import os
import torch
from torch import nn
from tqdm import tqdm
from lib.utils.config import cfg
from torch.utils.data import DataLoader
from lib.datasets.build_datasets import build_dataset
from lib.models.model_utils import init_net, copy_state_dict, get_scheduler

from .losses import GANLoss, GMMLogLoss, Sample_GMM, MaskedL1Loss, VGGLoss

from torch.cuda.amp import autocast, GradScaler

from lib.utils.util import tensor2im, save_image


class Trainer():
    def __init__(self, model, config=None) -> None:
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        
        self.model = model

        self.batch_size = self.cfg.batch_size
        self.Tensor = torch.cuda.FloatTensor
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.loss_mse = torch.nn.MSELoss().to(self.device)
        self.loss_gmm = GMMLogLoss(self.cfg.A2H_GMM_ncenter, self.cfg.A2H_GMM_ndim, self.cfg.A2H_GMM_sigma_min).to(self.device)
        self.loss_mask = MaskedL1Loss().to(self.device)
        self.loss_l1 = nn.L1Loss().to(self.device)
        self.loss_vgg = VGGLoss().to(self.device)
        self.loss_flow = nn.L1Loss().to(self.device)
        self.loss_gan = GANLoss(self.cfg.gan_mode, tensor=self.Tensor)
        
        if self.cfg.fp16:
            self.scaler = GradScaler()
        

        self.init_network()
        # print(self.model.f2f.FFG.state_dict()['netG.model.model.5.weight'][0,0,0,:2], 'x')
        # print(self.f2f_g.state_dict()['netG.model.model.5.weight'][0,0,0,:2], 'x')
        self.configure_optimizers()
        self.load_checkpoint()
        # print(self.model.f2f.FFG.state_dict()['netG.model.model.5.weight'][0,0,0,:2], 'g')
        # print(self.f2f_g.state_dict()['netG.model.model.5.weight'][0,0,0,:2], 'g')
        
        # if self.cfg.train.write_summary:
        #     from torch.utils.tensorboard import SummaryWriter
        #     self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))
    
    def set_requries_grad(self, net, requires_grad=False):
        if net is not None:
            for p in net.parameters():
                p.requires_grad = requires_grad
        
    def configure_optimizers(self):
        if self.cfg.task == 'Audio2Feature':
            # self.opt = torch.optim.Adam([{'params':self.model.a2f.parameters(), 'initial_lr': self.cfg.lr}], lr=self.cfg.lr, betas=(0.9, 0.99))
            self.opt = torch.optim.SGD([{'params':self.model.a2f.parameters(), 'initial_lr': self.cfg.lr}], lr=self.cfg.lr, momentum=0.9, nesterov=True)
        elif self.cfg.task == 'Audio2Headpose':
            self.opt = torch.optim.Adam([{'params':self.model.a2h.parameters(), 'initial_lr': self.cfg.lr}], lr=self.cfg.lr, betas=(0.9, 0.99))
        else:
            self.opt_g = torch.optim.Adam([{'params':self.model.f2f.FFG.parameters(), 'initial_lr':self.cfg.lr}], lr=self.cfg.lr, betas=(0.5, 0.99))
            self.opt_d = torch.optim.Adam([{'params':self.model.f2f.FFD.parameters(), 'initial_lr':self.cfg.lr}], lr=self.cfg.lr, betas=(0.5, 0.999))
            
            self.schduler_d = get_scheduler(self.opt_d, self.cfg)
            self.schduler_g = get_scheduler(self.opt_g, self.cfg)
            
    def init_network(self):
        if self.cfg.task == 'Audio2Feature':
            self.a2f = init_net(self.model.a2f, init_type='normal', init_gain=0.02, gpu_ids=['cuda:0'])
        elif self.cfg.task == 'Audio2Headpose':
            self.a2h = init_net(self.model.a2h, init_type='normal', init_gain=0.02, gpu_ids=['cuda:0'])
        else:
            print('Task is:', self.cfg.task)
            self.f2f_g = init_net(self.model.f2f.FFG, init_type='normal', init_gain=0.02, gpu_ids=['cuda:0'])
            self.f2f_d = init_net(self.model.f2f.FFD, init_type='normal', init_gain=0.02, gpu_ids=['cuda:0'])

    def load_checkpoint(self):
        model_dict = self.model.model_dict()
        # resume training, including model weight, opt, steps
        if os.path.exists(self.cfg.checkpoints):
            checkpoints = torch.load(self.cfg.checkpoints)
            self.weights_epoch = checkpoints['epoch']
        # load model weights only
        elif os.path.exists(self.cfg.pretrained_modelpath):
            checkpoints = torch.load(self.cfg.pretrained_modelpath)
            for key in model_dict.keys():
                if key in checkpoints.keys():
                    copy_state_dict(model_dict[key], checkpoints[key])
            self.weights_epoch = checkpoints['epoch']
        else:
            self.weights_epoch = 0
            
    def backward_d(self, input_feature_maps, tgt_image, fake_pred):
        '''
        Calculate GAN loss for the discriminator
        '''
        real_ab = torch.cat((input_feature_maps, tgt_image), dim=1)
        fake_ab = torch.cat((input_feature_maps, fake_pred), dim=1)
        
        pred_real = self.f2f_d(real_ab)
        pred_fake = self.f2f_d(fake_ab.detach())
        
        with autocast():
            loss_d_real = self.loss_gan(pred_real, True) * 2
            loss_d_fake = self.loss_gan(pred_fake, False)
        
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        
        if not self.cfg.fp16:
            loss_d.backward()
        else:
            self.scaler.scale(loss_d).backward()
        return loss_d
            
    def backward_g(self, input_feature_maps, tgt_image, fake_pred):
        '''
        Calculate GAN and other loss for the generator
        '''
        real_ab = torch.cat((input_feature_maps, tgt_image), dim=1)
        fake_ab = torch.cat((input_feature_maps, fake_pred), dim=1)
        pred_real = self.f2f_d(real_ab)
        pred_fake = self.f2f_d(fake_ab)
        
        loss_g = self.loss_gan(pred_fake, True)
        loss_l1 = self.loss_l1(fake_pred, tgt_image) * self.cfg.lambda_L1  # Color
        loss_vgg, loss_style = self.loss_vgg(fake_pred, tgt_image, style=True)
        loss_vgg = torch.mean(loss_vgg) * self.cfg.lambda_feat
        loss_style = torch.mean(loss_style) * self.cfg.lambda_feat
        
        # GAN feature matching loss
        def compute_FeatureMatching_loss(pred_fake, pred_real, cfg):
            loss_FM = torch.zeros(1).cuda()
            feat_weights = 4.0 / (cfg.n_layers_D + 1)
            D_weights = 1.0 / cfg.num_D
            for i in range(min(len(pred_fake), cfg.num_D)):
                for j in range(len(pred_fake[i])):
                    loss_FM += D_weights * feat_weights * \
                        self.loss_l1(pred_fake[i][j], pred_real[i][j].detach()) * cfg.lambda_feat
            
            return loss_FM
        
        loss_fm = compute_FeatureMatching_loss(pred_fake, pred_real, self.cfg)
        
        if not self.cfg.fp16:
            loss_gg = loss_g + loss_l1 + loss_vgg + loss_style + loss_fm
            loss_gg.backward()
        else:
            with autocast():
                loss_gg = loss_g + loss_l1 + loss_vgg + loss_style + loss_fm
            self.scaler.scale(loss_gg).backward()
            
        return loss_gg
    
    def training_step(self, batch, epoch, step):
        losses = {}
        opdict = {}
        
        if self.cfg.task == 'Audio2Feature':
            self.model.a2f.train()

            audio_feats, target_info = batch
            audio_feats = audio_feats.to(self.device)
            target_info = target_info.to(self.device)
            
            preds = self.model.a2f(audio_feats)  # [36, 240, 204]
            frame_future = self.cfg.frame_future
            mouth_index = list(range(48, 60))
            if not frame_future == 0:
                loss_mse = self.loss_mse(preds[:, frame_future:], target_info[:, :-frame_future]) * 1000
                # loss_mouth = self.loss_mse(preds[:, frame_future:], target_info[:, :-frame_future]) * 1000
            else:
                loss_mse = self.loss_mse(preds, target_info) * 1000
            
            losses['loss_mse'] = loss_mse
            # losses['loss_mouth'] = loss_mouth
        
        elif self.cfg.task == 'Audio2Headpose':
            headpose_audio_feats, history_headpose, target_headpose = batch      
            headpose_audio_feats = headpose_audio_feats.to(self.device)
            history_headpose = history_headpose.to(self.device)
            target_headpose = target_headpose.to(self.device)
            
            if self.cfg.audio_windows == 2:
                bs, item_len, ndim = headpose_audio_feats.shape
                headpose_audio_feats = headpose_audio_feats.reshape(bs, -1, ndim * 2)
            else:
                bs, item_len, _, ndim = headpose_audio_feats.shape
            
            preds_headpose = self.model.a2h.forward(history_headpose, headpose_audio_feats)
            
            loss_gmm = self.loss_gmm(preds_headpose, target_headpose)
            
            if not self.cfg.smooth_loss == 0:
                mu_gen = Sample_GMM(preds_headpose, self.model.a2h.WaveNet.ncenter, self.model.a2h.WaveNet.ndim, sigma_scale=0)
                loss_smooth = (mu_gen[:, 2:] + target_headpose[:, :-2] - 2 * target_headpose[:, 1:-1]).mean(dim=2).abs().mean()
                losses['loss_smooth'] = loss_smooth * self.cfg.smooth_loss
                
            losses['loss_gmm'] = loss_gmm
        else:
            feature_map, cand_img, tgt_img, facial_mask = batch['feature_map'], batch['cand_image'], batch['tgt_image'], batch['weight_mask']
            
            feature_map = feature_map.to(self.device)
            cand_img = cand_img.to(self.device)
            tgt_img = tgt_img.to(self.device)
            facial_mask = facial_mask.to(self.device)
            
            input_feature_maps, fake_pred = self.model.f2f.forward(feature_map, cand_img)
            
            # update D
            self.set_requries_grad(self.f2f_d, True)
            self.opt_d.zero_grad()
            if (epoch + 1) % 10 == 0:
                print(self.opt_d.state_dict()['param_groups'][0]['lr'])
            if not self.cfg.fp16:
                loss_d = self.backward_d(input_feature_maps, tgt_img, fake_pred)
                self.opt_d.step()
            else:
                with autocast():
                    loss_d = self.backward_d(input_feature_maps, tgt_img, fake_pred)
                    self.opt_d.step()
                self.scaler.step(self.opt_d)
            
            self.schduler_d.step()
                
            losses['loss_d'] = loss_d
                
            # update G
            self.set_requries_grad(self.f2f_g, True)
            self.opt_g.zero_grad()
            if (epoch + 1) % 10 == 0:
                print(self.opt_g.state_dict()['param_groups'][0]['lr'])

            if not self.cfg.fp16:
                loss_gg = self.backward_g(input_feature_maps, tgt_img, fake_pred)
                self.opt_g.step()
            else:
                with autocast():
                    loss_gg = self.backward_g(input_feature_maps, tgt_img, fake_pred)
                    self.opt_g.step()
                self.scaler.step(self.opt_g)
                self.scaler.update()
            self.schduler_g.step()
            
            losses['loss_gg'] = loss_gg
            if step % 5000 == 0:
                # save the middle generation image
                pred_fake = tensor2im(fake_pred[0])
                if not os.path.exists(self.cfg.gan_sample_dir):
                    os.makedirs(self.cfg.gan_sample_dir, exist_ok=True)
                save_image(pred_fake, os.path.join(self.cfg.gan_sample_dir, f'{epoch}_{step}_generation.png'))
            
        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        
        return losses, opdict
    
    def validation_step(self):
        self.model.a2f.eval()
        
    def evaluate(self):
        pass
    
    def prepra_data(self):
        '''
        build dataset and return DataLoader
        '''
        self.train_dataset = build_dataset(self.cfg)
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        
    def fit(self):
        self.prepra_data()
        # resume 
        self.load_checkpoint()
        save_dir = os.path.join(self.cfg.output_dir, self.cfg.task)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        f = open(self.cfg.loss_file, 'a+')
        start_epoch = self.weights_epoch
        if self.cfg.task != 'Feature2Face':
            for epoch in tqdm(range(start_epoch, self.cfg.train.max_epochs)):
                print(f'Training epoch {epoch}')
                for step, batch in enumerate(tqdm(self.train_dataloader)):
                    losses, opdict = self.training_step(batch, epoch, step)
                    all_loss = losses['all_loss']
                    self.opt.zero_grad()
                    all_loss.backward()
                    self.opt.step()
                    
                if epoch > 0 and epoch % self.cfg.train.checkpoint_steps == 0:
                    f.write(f'loss of epoch {epoch} is : {all_loss.item()}\n')
                    print(all_loss.item())
                    model_dict = self.model.model_dict()
                    model_dict['opt'] = self.opt.state_dict()
                    model_dict['epoch'] = epoch
                    model_dict['batch_size'] = self.batch_size
                    torch.save(model_dict, os.path.join(save_dir, f'{self.cfg.ck_add_info}_{epoch}.tar'))
          
        else:
            for epoch in tqdm(range(start_epoch, self.cfg.train.max_epochs)):
                print(f'Training epoch {epoch}')
                for step, batch in enumerate(tqdm(self.train_dataloader)):
                    losses, opdict = self.training_step(batch, epoch, step)
                    loss_gg = losses['loss_gg'].item()
                    loss_d = losses['loss_d'].item()
                
                if epoch > 0 and epoch % self.cfg.train.checkpoint_steps == 0:
                    f.write(f'loss of epoch {epoch} is : {loss_gg}\n')
                    f.write(f'loss of epoch {epoch} is : {loss_d}\n')
                    print(f'loss of epoch {epoch}','D: ',  loss_d, 'G:', loss_gg, '\n')
                    model_dict = {}
                    model_dict['opt_g'] = self.opt_g.state_dict()
                    model_dict['opt_d'] = self.opt_d.state_dict()
                    
                    model_dict['epoch'] = epoch
                    model_dict['batch_size'] = self.batch_size
                    
                    model_dict['feature2face_g'] = self.f2f_g.state_dict()
                    model_dict['feature2face_d'] = self.f2f_d.state_dict()
                    
                    torch.save(model_dict, os.path.join(save_dir, f'{self.cfg.ck_add_info}_{epoch}.tar'))
                        
        f.close()
