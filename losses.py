
import torch
import torch.nn as nn
import torch.nn.functional as F
def sparse_softmax(x,dim=1, k=10):
    """
    Compute the sparse softmax of input tensor x, only keeping the top k values.

    Args:
        x: Input tensor with shape (batch_size, num_classes).
        k: The number of values to keep after the softmax operation.

    Returns:
        Tensor with the same shape as x, but with values outside the top k set to zero.
    """


    # Sort the softmax probabilities and keep only the top k values
    top_k_values, top_k_indices = torch.topk(x, k=k, dim=dim)

    # Create a mask where values outside the top k are set to zero
    mask = torch.zeros_like(x)
    mask.scatter_(dim, top_k_indices, 1)

    # Apply the mask to the softmax probabilities to create the sparse softmax
    sparse_x = x * mask + (1-mask)*(-1e6)
    # Compute the standard softmax of x
    sparse_softmax_x = torch.softmax(sparse_x, dim=dim)
    return sparse_softmax_x, mask

def CE(output, target,ignore_index=-100):
    '''
    Output: (B,L,C)。未经过softmax的logits,C代表类别数
    Target: (B,L)
    '''
    output = output.reshape(-1, output.shape[-1])  # (*,C)
    target = target.reshape(-1).long()  # (*)
    return nn.CrossEntropyLoss(ignore_index=ignore_index,label_smoothing=0.1)(output, target) #默认size_average=True，会把B*L所有词loss平均

def soft_label_loss(logits, target, ignore_index=-100):
    '''
    logits: (B,L,C)。未经过softmax的logits,C代表类别数
    target: (B,L)
    ignore_index: target中需要跳过计算loss的idx
    '''
    ignored_mask = torch.ones_like(target).float()
    ignored_mask[target.eq(ignore_index)] = 0
    
    soft_labels,_ = convert_to_soft_label(target,num_classes=logits.shape[-1])
    logits = logits.reshape(-1, logits.shape[-1])  # (*,C)
    soft_labels = soft_labels.reshape(-1, soft_labels.shape[-1])
    ignored_mask = ignored_mask.reshape(-1,1)
    log_softmax = F.log_softmax(logits, dim=-1)
    
    loss = -torch.sum(soft_labels * log_softmax * ignored_mask) / torch.sum(ignored_mask)
    return loss

def sparse_soft_label_loss(logits, target, ignore_index=-100):
    '''
    logits: (B,L,C)。未经过softmax的logits,C代表类别数
    target: (B,L)
    ignore_index: target中需要跳过计算loss的idx
    '''
    ignored_mask = torch.ones_like(target).float()
    ignored_mask[target.eq(ignore_index)] = 0
    
    soft_labels,_ = convert_to_soft_label(target,num_classes=logits.shape[-1])
    logits = logits.reshape(-1, logits.shape[-1])  # (*,C)
    soft_labels = soft_labels.reshape(-1, soft_labels.shape[-1])
    ignored_mask = ignored_mask.reshape(-1,1)
    
    sparse_softmax_x,sparse_mask = sparse_softmax(logits,dim=1)
    log_softmax = sparse_mask*((sparse_softmax_x + 1e-6).log())
    
    loss = -torch.sum(soft_labels * log_softmax * ignored_mask) / torch.sum(ignored_mask)
    return loss

def comput_R_drop_loss(p,q, target, ignore_index=-100):
    ignored_mask = torch.ones_like(target).float()
    ignored_mask[target.eq(ignore_index)] = 0
    ignored_mask = ignored_mask.reshape(-1,1)
    p = p.reshape(-1, p.shape[-1])
    q = q.reshape(-1, q.shape[-1])

    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    loss = torch.sum((p_loss+q_loss)*ignored_mask/2)/torch.sum(ignored_mask)
    return loss
    
def comput_KL_loss(p,q,target,ignore_index=-100):
    ignored_mask = torch.ones_like(target).float()
    ignored_mask[target.eq(ignore_index)] = 0
    ignored_mask = ignored_mask.reshape(-1,1)
    p = p.reshape(-1, p.shape[-1])
    q = q.reshape(-1, q.shape[-1])

    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')


    loss = torch.sum(p_loss*ignored_mask)/torch.sum(ignored_mask)
    return loss
def convert_to_soft_label(target, num_classes,alpha=0.9, beta = None):
    """
    Convert a long vector to a soft label tensor.

    Args:
        target (torch.LongTensor): A tensor of shape (B, L) containing long type labels.
        num_classes (int): The number of classes.

    Returns:
        soft_label (torch.FloatTensor): A tensor of shape (B, L, C) containing soft labels.
    """
    B, L = target.size()
    if beta is None:
        beta = (1-alpha)/(num_classes-1)
    hard_label = torch.zeros(B, L, num_classes, dtype=torch.float).to(target.device)
    hard_label.scatter_(2, target.unsqueeze(-1), 1)
    soft_label = hard_label*alpha + (1-hard_label)*beta
    
    return soft_label,hard_label
def BCE(output, target,ignore_index=-100):
    '''
    Output: (B,L,C)。未经过softmax的logits,C代表类别数
    Target: (B,L)
    '''
    B, L = target.size()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    label = torch.zeros(B, L, output.shape[-1], dtype=torch.float).to(target.device)
    label.scatter_(2, target.unsqueeze(-1), 1)
    
    ignored_mask = torch.ones_like(target).float()
    ignored_mask[target.eq(ignore_index)] = 0
    
    output = output.reshape(-1, output.shape[-1])  # (*,C)
    label = label.reshape(-1, label.shape[-1])  # (*,C)
    ignored_mask = ignored_mask.reshape(-1,1)
    loss = criterion(output,label)
    loss = torch.sum(loss*ignored_mask)/torch.sum(ignored_mask)
    return loss

def FocalLoss(logits, target,ignore_index=-100, gamma=2):
    '''
    logits: (B,L,C)。未经过softmax的logits,C代表类别数
    target: (B,L)
    ignore_index: target中需要跳过计算loss的idx
    '''
    ignored_mask = torch.ones_like(target).float()
    ignored_mask[target.eq(ignore_index)] = 0
    
    soft_labels,hard_labels = convert_to_soft_label(target,num_classes=logits.shape[-1])
    logits = logits.reshape(-1, logits.shape[-1])  # (*,C)
    soft_labels = soft_labels.reshape(-1, soft_labels.shape[-1])
    hard_labels = hard_labels.reshape(-1, hard_labels.shape[-1])
    
    ignored_mask = ignored_mask.reshape(-1,1)
    log_softmax = F.log_softmax(logits, dim=-1)
    
    pt = hard_labels*F.softmax(logits,dim=-1)#B,C

    loss = -torch.sum((1-pt)**gamma*soft_labels * log_softmax *ignored_mask) / torch.sum(ignored_mask)
    return loss

def simcse_unsup_loss(y_pred,lamda=0.05):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]

    """
    y_true = torch.arange(y_pred.shape[0]).to(y_pred.device)
    y_true = y_true - y_true//(y_pred.shape[0]//2)*(y_pred.shape[0]//2) + (y_pred.shape[0]//2)*(1-y_true//(y_pred.shape[0]//2))
    # y_true = (y_true - y_true % 2 * 2) + 1
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    
    sim = sim - torch.eye(y_pred.shape[0]).to(y_pred.device) * 1e12
    sim = sim / lamda

    loss = F.cross_entropy(sim, y_true.long())
    return loss