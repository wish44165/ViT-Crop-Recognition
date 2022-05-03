import torch.optim as optim

def create_optim(cfg, model):
    cfg_optim = cfg['train']['optimizer']
    assert cfg_optim['mode'] in ['adamW', 'adam'], '{} havn\'t implemented'.format(cfg_optim['mode'])
    if cfg_optim['mode'] == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=cfg_optim['lr'], weight_decay=cfg_optim['weight_decay'])
    elif cfg_optim['mode'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg_optim['lr'], weight_decay=cfg_optim['weight_decay'])
    return optimizer