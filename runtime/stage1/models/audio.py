from models.htsat import HTSATWrapper
from models.cnn14 import CNN14Wrapper
from models.ced import CEDSmallWrapper
from models.beats import BEATsBaseWrapper

def get_audio_encoder(name: str):
    if name == "HTSAT":
        return HTSATWrapper, 768
    elif name == "Cnn14":
        return CNN14Wrapper, 2048
    elif name in {"CEDSmall", "CED-Small", "CED_SMALL", "CED"}:
        return CEDSmallWrapper, 384
    elif name in {"BEATsBase", "BEATs", "BEATSBase", "BEATS", "BEATs-Base"}:
        return BEATsBaseWrapper, 768
    else:
        raise Exception('The audio encoder name {} is incorrect or not supported'.format(name))
