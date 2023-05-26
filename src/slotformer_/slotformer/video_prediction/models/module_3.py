import pdb
import torch
import torchvision
import math
import numpy as np

from torch.nn import ReLU, Sigmoid, Linear, Sequential, LayerNorm, Module
from torch.nn import Conv2d, MaxPool2d, ReLU, ConvTranspose2d, BatchNorm2d
from torch.nn import Softmax, Sigmoid, MSELoss, Parameter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import one_hot, affine_grid, grid_sample

class PositionEncoder(Module):

    def __init__(self, args):
        super(PositionEncoder, self).__init__()
        self.mlp = Sequential(
            Linear(2, args.encoder.output_dim),
            ReLU(),
            Linear(args.encoder.output_dim, args.encoder.output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

class PositionEmbedding(Module):

    def __init__(self, args):
        super(PositionEmbedding, self).__init__()
        #the position dimension
        self.dim = args.encoder.output_dim

        #the embeddings
        #pe: MAX_LEN x DIM
        self.pe = torch.zeros(args.transition.max_len,self.dim)
        
        #seq: DIM
        #pos: MAX_LEN
        seq = torch.exp(-1*torch.arange(0,self.dim,2)/float(self.dim)*math.log(10000))
        pos = torch.arange(0,args.transition.max_len)
        
        #pe: 1 x MAX_LEN x 1 x DIM
        self.pe[:,0::2] = torch.sin(pos.unsqueeze(-1)*seq.unsqueeze(0))
        self.pe[:,1::2] = torch.cos(pos.unsqueeze(-1)*seq.unsqueeze(0))
        self.pe = self.pe.unsqueeze(0)
        self.pe = self.pe.unsqueeze(2)

        self.pe = self.pe.to(args.device)
    
    def forward(self, x, off = 0):
        '''
        Args:
            x: B x SEQ_LEN x NOB x D
            off: starting position
        Returns:
            x: B x SEQ_LEN x NOB x D
        '''
        x = x + self.pe[:,off: off+x.shape[1],:]
        return x



'''Encoding the masks to latent states
    Given the mask rcnn masks this module will try  to get the
    object latents from the mask rcnn output
    should i use the transformer encoder?
'''
class RCNNMasks(Module):

    def __init__(self,args):
        super(RCNNMasks, self).__init__()
        self.args = args
        self.objects = self.args.objects
        
        #get the model
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True,num_classes=91)
        # self.model.requires_grad = False
        
        #the image size
        self.image_size = (self.args.height, self.args.width)
        self.blank_mask = torch.zeros(self.image_size).unsqueeze(0).unsqueeze(0).cuda()
        self.blank_mask = [self.blank_mask for i in range(self.objects)]

    def forward(self, x):
        '''
        Args:
            x: B x 3 x H x W
        Output:
            masks: B x NO_OB x H x W
        '''
        self.model.eval()
        self.blank_mask = [i.cuda() for i in self.blank_mask]

        with torch.no_grad():
            image_list = [i for i in x]
            output = self.model(image_list)
            masks = []
            for i in output:
                masks += torch.cat([i['masks'][:self.objects]] + self.blank_mask, dim = 0)[:self.objects,...].unsqueeze(0)
            
            #masks will be list of masks
            masks = torch.cat(masks, dim = 0)
            return masks


class RCNNEncoder2(Module):

    def __init__(self, args):
        super(RCNNEncoder2, self).__init__()
        
        #the new args
        self.args = args
        self.internal_channels = self.args.rcnn_encoder.internal_channels
        self.linear_hidden_units = self.args.rcnn_encoder.hidden_units
        self.output_dim = self.args.rcnn_encoder.output_dim

        #the convolution
        self.conv1 = Conv2d(3, self.internal_channels, kernel_size = (3,3), padding = 1)
        self.conv2 = Conv2d(self.internal_channels, self.internal_channels, kernel_size = (3,3), padding = 1)
        
        self.conv3 = Conv2d(self.internal_channels, self.internal_channels, kernel_size = (3,3), padding = 1)
        self.conv4 = Conv2d(self.internal_channels, self.internal_channels, kernel_size = (3,3), padding = 1)
        
        self.conv5 = Conv2d(self.internal_channels, self.internal_channels, kernel_size = (3,3), padding = 1)
        self.conv6 = Conv2d(self.internal_channels, self.internal_channels, kernel_size = (3,3), padding = 1)
         
        self.conv7 = Conv2d(self.internal_channels, self.internal_channels, kernel_size = (3,3), padding = 1)
        self.conv8 = Conv2d(self.internal_channels, self.internal_channels, kernel_size = (3,3), padding = 1)
        
        #the linear layer
        image_size = self.args.height*self.args.width
        linear_layer_input_dim = (image_size // 64)*self.internal_channels
        self.linear_layer1 = Linear(linear_layer_input_dim, self.linear_hidden_units)
        self.linear_layer2 = Linear(self.linear_hidden_units, self.output_dim)

        #the activation function
        self.relu = ReLU(inplace = True)
        self.max_pool = MaxPool2d(kernel_size= (2,2),stride=2)

    def forward(self, mask):
        '''
        Args:
            masks: (B,1,H,W)
        Return:
            encoded_states: (B, ENCODING_DIM)
        '''
        output = self.conv1(mask)
        output = self.relu(output)
        output = self.conv2(output)
        
        output = self.max_pool(output)

        output = self.conv3(output)
        output = self.relu(output)
        output = self.conv4(output)
        
        output = self.max_pool(output)

        output = self.conv5(output)
        output = self.relu(output)
        output = self.conv6(output)
        
        output = self.max_pool(output)

        output = output.flatten(-3,-1)
        
        output = self.linear_layer1(output)
        output = self.relu(output)
        output = self.linear_layer2(output)

        return output


class RCNNEncoder(Module):

    def __init__(self, args):
        super(RCNNEncoder, self).__init__()
        
        #the new args
        self.args = args
        self.linear_hidden_units = self.args.rcnn_encoder.hidden_units
        self.output_dim = self.args.rcnn_encoder.output_dim
        self.num_heads = self.args.rcnn_encoder.num_heads
        self.num_layers = self.args.rcnn_encoder.num_layers
        self.input_dim = self.args.rcnn_encoder.input_dim

        #the mapping to lower dimension
        self.linear_layer1 = Linear(self.input_dim, self.linear_hidden_units)
        self.linear_layer2 = Linear(self.linear_hidden_units, self.output_dim)

        #the transformer encoder
        encoder_layers = TransformerEncoderLayer(self.output_dim, self.num_heads, (int) (self.output_dim / self.num_heads), dropout = 0.1, batch_first = True) 
        self.transformer = TransformerEncoder(encoder_layers, self.num_layers)

        #the attention mask (it is additive mask)
        self.mask = torch.zeros(self.args.objects + 1, self.args.objects + 1).cuda()

    def forward(self, features):
        '''
        Args:
            masks: (B, NUM_OF_OBJECTS+1, F_DIM)
        Return:
            encoded_states: (B,NUM_OF_OBJECTS+1, F_DIM)
        '''
        output = self.linear_layer1(features)
        output = self.linear_layer2(output)
        output = self.transformer(output, self.mask)
        return output


class SpatialBroadcastDecoder(Module):

    def __init__(self, args):
        super(SpatialBroadcastDecoder, self).__init__()
        
        #get the args
        self.args = args

        #get the input channels
        self.in_channels = self.args.rcnn_encoder.output_dim + 2
        self.hidden_channels = self.args.decoder.hidden_channels
        self.output_channels = 3 + 1

        #positional info
        self.input_grid_size = [self.args.height//8, self.args.width//8]
        position_x = torch.arange(-1,1,2/self.input_grid_size[1])
        position_y = torch.arange(-1,1,2/self.input_grid_size[0])
        
        pos_x, pos_y = torch.meshgrid(position_x, position_y)
        self.position_grid = torch.cat([pos_x.unsqueeze(0), pos_y.unsqueeze(0)], dim=0)
        self.position_grid = self.position_grid.permute(0,1,2)
        self.position_grid = self.position_grid.unsqueeze(0).to(self.args.device)

        #increase the size by 8 folds
        self.conv1 = ConvTranspose2d(self.in_channels, self.hidden_channels, (5,5), (2,2), padding = 2, output_padding = 1)
        self.conv2 = ConvTranspose2d(self.hidden_channels, self.hidden_channels, (5,5), (2,2), padding = 2, output_padding = 1)
        self.conv3 = ConvTranspose2d(self.hidden_channels, self.hidden_channels, (5,5), (2,2), padding = 2, output_padding = 1)

        #the final two convs
        self.conv5 = ConvTranspose2d(self.hidden_channels, self.output_channels+1, (5,5), (1,1), padding = 2)
        self.conv6 = ConvTranspose2d(self.output_channels+1, self.output_channels, (3,3), (1,1), padding = 1)
                
        #activation
        self.relu = ReLU()

    def forward(self, x):
        '''
        Args:   
            x: B x NO+1 x INPUT_DIM) tensor 
        Returns:
            output: (B x (C+1) x 8H x 8W) image
        '''
        #broadcast x to higher dimensions
        self.position_grid = self.position_grid.cuda()
        no_of_masks = x.shape[1]
        x = x.flatten(0,1)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.expand(-1,-1,self.input_grid_size[0],self.input_grid_size[1])
        
        position = self.position_grid.expand(x.shape[0], -1, -1, -1)
        x = torch.cat((x,position), dim = 1)

        output = self.conv1(x)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.relu(output)

        output = self.conv5(output)
        output = self.relu(output)
        output = self.conv6(output)

        output = output.view(-1, no_of_masks, output.shape[1], output.shape[2], output.shape[3])
        return output

'''
Transition: transformer with attention to previous states.
'''
class Transition(Module):

    def __init__(self,args):
        super(Transition,self).__init__()
        
        #meta data
        self.args = args
        self.num_layers = self.args.transition.num_layers
        self.num_heads = self.args.transition.num_heads
        self.dim = args.encoder.output_dim
        self.objects = args.objects
        self.history = args.history
        self.device = args.device

        #the position embedder
        self.position_embedings = PositionEmbedding(args)
        
        #encdoder layers
        encoder_layers = TransformerEncoderLayer(self.dim, self.num_heads, (int) (self.dim / self.num_heads), dropout = 0.1, batch_first = True) 
        #, batch_first = True not supported on current batch
        self.transformer = TransformerEncoder(encoder_layers, self.num_layers)
        
        #the mask
        self.mask = self.get_masks()

    def get_masks(self):
        #masks: h x h
        masks = torch.triu(torch.ones(self.history, self.history) - float('inf'), diagonal = 1)
        #masks = masks.repeat(self.objects, self.objects)
        masks = masks.unsqueeze(-1)
        masks = masks.unsqueeze(1)
        masks = masks.repeat(1,self.objects,1,self.objects)
        masks = masks.reshape(self.objects*self.history,-1)
        masks = masks.to(self.device)
        return masks

    def forward(self, x, offset = 0):
        '''
        Args:
            x: [B x SEQ_LEN] x NOB x DIM 
        Return:
            update: [B x NOB] x DIM 
        '''
        #check for change in history len
        if(x.shape[1] != self.history):
            self.history = x.shape[1]
            self.mask = self.get_masks()

        #return the ans
        x = self.position_embedings(x)
        
        #x: (B x [SEQ_LEN x NOB] x DIM 
        x = x.view(-1, self.objects*self.history, self.dim)
        
        #change to seq first
        x = self.transformer(x, self.mask)

        #return the last objects
        x = x.unflatten(1, (self.history, self.objects))
        return x

        
'''Main module contains
    1. unet
    2. gaussian maks
    3. encoder
    4. transistion model
    5. decoder
'''
class Contrastive(Module):

    def __init__(self, args):
        super(Contrastive,self).__init__()
        
        #meta data image
        self.args = args
        self.height = args.height
        self.width = args.width
        self.channel = args.channels
        
        #meta data history rollout
        self.history = args.history
        self.rollout = None
        
        #meta data object and encoding dim
        self.no_of_objects = args.objects
        self.encoding_dim = args.encoder.output_dim
        self.batch_size = args.train.batch_size
        self.device = args.device
        self.hinge = 1

        #modules
        self.transition = Transition(self.args)
        self.static_selector = Parameter((torch.rand(self.args.encoder.output_dim, requires_grad=True)-0.5))
        self.sigmoid = Sigmoid()

        #some useful layers
        self.mse_loss = MSELoss(reduction='sum')
        self.softmax = Softmax(dim = 1)

        #to get the position
        self.y_index = torch.arange(self.args.height).float().unsqueeze(1)
        self.y_index = self.y_index.to(self.device)
        self.x_index = torch.arange(self.args.width).float().unsqueeze(0)
        self.x_index = self.x_index.to(self.device)
        
    def get_position(self, masks):
        '''
        Args:
            masks: * x H x W
        Returns:
            position: * x 2
        '''
        self.y_index = self.y_index.cuda()
        self.x_index = self.x_index.cuda()
        mean_y = torch.mean(masks*self.y_index, (-1,-2))
        mean_x = torch.mean(masks*self.x_index, (-1,-2))
        position = torch.cat([mean_x.unsqueeze(-1), mean_y.unsqueeze(-1)], -1)
        return position
    
    # def energy(self, state,next_state, no_trans=False):
    #     """Energy function based on normalized squared L2 norm."""

    #     norm = 0.5 / (self.sigma**2)
    #     diff = state - next_state
    #     return norm * diff.pow(2).sum(2).mean(1)


    def mask_max_val_loss(self, curr_masks):
        '''
        compute the max value of mask and ground it to
        less than 1
        Args:
            curr_masks: B x N x H x W
        Output:  
            mmloss: float (MSE avergaed over B)
        '''
        batch_size = curr_masks.size(0)
        objs_flat = curr_masks.reshape((batch_size, self.no_of_objects, -1))
        curr_mask_max = torch.max(objs_flat, dim=2).values
        zero_tensor = torch.zeros(curr_mask_max.shape).to(self.device)
        curr_pixel_loss = (torch.max(zero_tensor,1 - curr_mask_max)).mean()
        return curr_pixel_loss

    def contrastive_loss(self, pred_next, next_state):
        '''
        compute contrastive loss between predicted latent and
        prev latent + update. Also negative samples are sampled to
        pass negative loss for both z_t and z_t+1
        Args:
            state(z_t): B x N x {embedding_dimension}
            pred_next(z_t+1 pred): B x N x {embedding_dimension}
            next_state(z_t+1): B x N x {embedding_dimension}
        Output:  
            closs: float (MSE avergaed over B)
        '''
        perm = np.random.permutation(pred_next.shape[0]) 
        neg_state = next_state[perm] # Generate a random permutation of next states present in the batch
        
        #d(z_t + update, z_t+1)
        self.pos_loss = ((pred_next-next_state)**2).sum((-1,-2,-3))
        
        #max(0, hinge - d(z_t+1, z'_t+1))
        zeros = torch.zeros_like(self.pos_loss)
        self.neg_loss = torch.max(zeros, self.hinge -  ((pred_next-neg_state)**2).sum((-1,-2,-3))).mean()
        loss = self.pos_loss + self.neg_loss # Total contrastive Loss
        return loss
    

    def decoder_loss(self, next_obs, curr_obs):
        '''
        compute mse between original image and constructed image
        Args:
            next_obs: B x C x H x W
            curr_obs: B x C x H x W
        Output:  
            dloss: float (MSE avergaed over B)
        '''
        dloss = ((curr_obs - next_obs)**2).sum((-3,-2,-1))
        dloss = dloss.mean()
        return dloss


    def pair_loss(self, masks):
        '''
        computes pair loss of masks
        Args:
            masks: B x N x H x W
        Output:  
            ploss: float (avergaed over B)
        '''
        #take product of two objects at time
        loss = torch.tensor(0.0).to(self.device)
        for object_1 in range(0, self.no_of_objects):
            mask_object_1 = masks[:,object_1,:,:]
            for object_2 in range(object_1+1, self.no_of_objects):
                mask_object_2 = masks[:,object_2,:,:]
                prod = mask_object_1*mask_object_2
                loss += prod.sum((1,2)).mean()

        return loss 

    def reconstruct(self, masks):
        '''
        Args:
            masks: (B x NO_OF_OBJ+1 x 4 x H x W)
        Returns:
            images: (B x 3 x H x W)
        '''
        objects_content = masks[:,:,0:3,:,:]
        objects_mask = self.softmax(masks[:,:,3:,:,:])
        objects = objects_mask*objects_content
        images = torch.sum(objects, 1)
        return images, objects_mask

    def forward(self, input, mode = 'train', vis_samples = False):
        
        #the return dictionary
        results = dict()

        #set the rollout
        self.rollout = self.args[mode].rollout

        #get the history (B x His) x NoOb x Slot_dim
        input_slots = input['slots'][:,0:self.history,...]
        rollout_slots = input['slots'][:,self.history:,...]

        #change slots to dynamic and static part
        static_selector = self.sigmoid(self.static_selector)
        input_slots_static = input_slots*static_selector
        input_slots_dyn = input_slots - input_slots_static
        rollout_slots_static = rollout_slots*static_selector
        rollout_slots_dyn = rollout_slots - rollout_slots_static

        #the latest static part
        static_part = input_slots_static[:,-1,...]
        
        #reconstruct the original images
        decoder_loss = torch.tensor([0.0]).cuda()
        contrastive_loss = torch.tensor([0.0]).to(self.device)
        sanity_loss = -1*(static_selector**2).sum()
        pair_loss = torch.tensor([0.0]).to(self.device)
        vel_loss = torch.tensor([0.0]).to(self.device)
        
        #return slot list
        pred_slots = []
        
        #the prevslot
        prev_slots = input_slots[:,-1,...]
        
        for i in range(self.rollout):
            
            #encoded states B x HISTORY x NO_OBJ
            delta_future_slot_dyn = self.transition(input_slots_dyn)[:,-1,...]
            future_slots_dyn = input_slots_dyn[:,-1,...] + delta_future_slot_dyn
            
            #compute the final slots
            future_slots = future_slots_dyn*(1 - static_selector) + static_part*static_selector

            #loss term
            contrastive_loss += self.contrastive_loss(future_slots_dyn, rollout_slots_dyn[:,i,...])  
            #contrastive_loss += ((rollout_slots_static[:,i,...] - static_part)**2).sum((-3,-2,-1)).mean()
        
            #add current slots to the prediction and also to the input 
            input_slots_dyn = input_slots_dyn
            input_slots_dyn = torch.cat([input_slots_dyn[:,:-1,...], future_slots_dyn.unsqueeze(1)], dim = 1)
            prev_slots = future_slots

            #append the slots to list
            pred_slots.append(future_slots.unsqueeze(1))

            #append this information
            if(mode != 'train'):
                if(vis_samples):
                    rollout_generated.append(predicted_images[:sample_number])
                    rollout_masks.append(curr_masks[:sample_number])
        
        results = {
            "contrastive_loss":contrastive_loss, 
            "vel_loss": vel_loss, 
            "decoder_loss": decoder_loss, 
            "sanity_loss": sanity_loss,
            "pair_loss": pair_loss,
            "pred_slots": torch.cat(pred_slots, dim = 1)
        }

        positions = torch.zeros(input['slots'].shape[0], self.history, self.no_of_objects, 2).cuda()
        rollout_positions = torch.zeros(input['slots'].shape[0], self.rollout, self.no_of_objects, 2).cuda()
        rollout_velocities = rollout_positions
        
        #add the position and velocities
        if(mode != 'train'):
            #add positions
            results['history_positions'] = positions
            results['rollout_positions'] = rollout_positions
            results['rollout_velocities'] = rollout_velocities

        #add sample images
        if(vis_samples):
            
            #original images
            results['history_images'] = image_history[:sample_number,-1,...]
            results['rollout_images'] = image_rollout[:sample_number,...]

            #masks
            results['history_masks'] = masks[:sample_number,-1,...]
            results['rollout_masks'] = torch.cat([t.unsqueeze(1) for t in rollout_masks], dim = 1)

            #reconstruction and new image generated
            results['history_generated'] = reconstructed_images[:sample_number,-1,:]
            results['rollout_generated'] = torch.cat([t.unsqueeze(1) for t in rollout_generated], dim = 1)

            #if input has rcn masks then forward
            if('rcnn_masks' in input.keys()):
                results['rcnn_masks'] = input['rcnn_masks']

        #pdb.set_trace()
        return results    
  

    def generate(self, input, rollout):
        
        #the return dictionary
        results = dict()

        #set the rollout
        self.rollout = self.args[mode].rollout

        #get the history (B x His) x NoOb x Slot_dim
        input_slots = input['slots'][:,0:self.history,...]

        #change slots to dynamic and static part
        static_selector = self.sigmoid(self.static_selector)
        input_slots_static = input_slots*static_selector
        input_slots_dyn = input_slots - input_slots_static

        #the latest static part
        static_part = input_slots_static[:,-1,...]
        
        #reconstruct the original images
        decoder_loss = torch.tensor([0.0]).cuda()
        contrastive_loss = torch.tensor([0.0]).to(self.device)
        sanity_loss = -1*(static_selector**2).sum()
        pair_loss = torch.tensor([0.0]).to(self.device)
        vel_loss = torch.tensor([0.0]).to(self.device)
        
        #return slot list
        pred_slots = []
        
        #the prevslot
        prev_slots = input_slots[:,-1,...]
        
        for i in range(self.rollout):
            
            #encoded states B x HISTORY x NO_OBJ
            delta_future_slot_dyn = self.transition(input_slots_dyn)[:,-1,...]
            future_slots_dyn = input_slots_dyn[:,-1,...] + delta_future_slot_dyn
            
            #compute the final slots
            future_slots = future_slots_dyn*(1 - static_selector) + static_part*static_selector

            #loss term
            #contrastive_loss += ((rollout_slots_static[:,i,...] - static_part)**2).sum((-3,-2,-1)).mean()
        
            #add current slots to the prediction and also to the input 
            input_slots_dyn = input_slots_dyn
            input_slots_dyn = torch.cat([input_slots_dyn[:,:-1,...], future_slots_dyn.unsqueeze(1)], dim = 1)
            prev_slots = future_slots

            #append the slots to list
            pred_slots.append(future_slots.unsqueeze(1))

            #append this information
            if(mode != 'train'):
                if(vis_samples):
                    rollout_generated.append(predicted_images[:sample_number])
                    rollout_masks.append(curr_masks[:sample_number])
        
        results = {
            "contrastive_loss":contrastive_loss, 
            "vel_loss": vel_loss, 
            "decoder_loss": decoder_loss, 
            "sanity_loss": sanity_loss,
            "pair_loss": pair_loss,
            "pred_slots": torch.cat(pred_slots, dim = 1)
        }

        positions = torch.zeros(input['slots'].shape[0], self.history, self.no_of_objects, 2).cuda()
        rollout_positions = torch.zeros(input['slots'].shape[0], self.rollout, self.no_of_objects, 2).cuda()
        rollout_velocities = rollout_positions
        
        #add the position and velocities
        if(mode != 'train'):
            #add positions
            results['history_positions'] = positions
            results['rollout_positions'] = rollout_positions
            results['rollout_velocities'] = rollout_velocities

        #add sample images
        if(vis_samples):
            
            #original images
            results['history_images'] = image_history[:sample_number,-1,...]
            results['rollout_images'] = image_rollout[:sample_number,...]

            #masks
            results['history_masks'] = masks[:sample_number,-1,...]
            results['rollout_masks'] = torch.cat([t.unsqueeze(1) for t in rollout_masks], dim = 1)

            #reconstruction and new image generated
            results['history_generated'] = reconstructed_images[:sample_number,-1,:]
            results['rollout_generated'] = torch.cat([t.unsqueeze(1) for t in rollout_generated], dim = 1)

            #if input has rcn masks then forward
            if('rcnn_masks' in input.keys()):
                results['rcnn_masks'] = input['rcnn_masks']

        #pdb.set_trace()
        return results    
  


