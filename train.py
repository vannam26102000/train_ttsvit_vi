import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.characters import CharactersConfig
# from vn_characters.vn_characters import VieCharacters
# from formater.customformater import formatter,formatter2
import argparse

def ljspeech_vi(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the LJSpeech meta data file to TTS format
    https://keithito.com/LJ-Speech-Dataset/"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "ljspeech"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[1].lower()[:-1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


dataset_config = BaseDatasetConfig(
    # formatter="ljspeech",
    meta_file_train="metadata.csv",
    # meta_file_attn_mask=os.path.join(output_path, "../LJSpeech-1.1/metadata_attn_mask.txt"),
    path="/kaggle/input/data/vi_tts",
)


from TTS.tts.utils.text.characters import BaseCharacters
_pad = "<PAD>"
_eos = "<EOS>"
_bos = "<BOS>"
_blank = "<BLNK>"  # TODO: check if we need this alongside with PAD
_characters= 'abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ0123456789'
_punctuations = "!'(),-.:;? "


class VieCharacters(BaseCharacters):
    def __init__(self, characters: str = _characters, 
                 punctuations: str = _punctuations, 
                 pad: str = _pad, 
                 eos: str = _eos, 
                 bos: str = _bos, 
                 blank: str = _blank, 
                 is_unique: bool = False, 
                 is_sorted: bool = True) -> None:
        super().__init__(characters, punctuations, pad, eos, bos, blank, is_unique, is_sorted)


    def _create_vocab(self):
        return super()._create_vocab()

class Vie_CharacterConfig(CharactersConfig):
    
    characters_class = VieCharacters
    characters = VieCharacters.characters
    blank = VieCharacters.blank
    eos = VieCharacters.eos
    bos = VieCharacters.bos
    pad = VieCharacters.pad
    punctuations = VieCharacters.punctuations

vie_characters = VieCharacters()
vie_char_conf = CharactersConfig(
                                     pad=vie_characters.pad,
                                     eos=vie_characters.eos,
                                     bos=vie_characters.bos,
                                     blank=vie_characters.blank,
                                     punctuations=vie_characters.punctuations,
                                     characters=vie_characters.characters)


audio_config = VitsAudioConfig(sample_rate=16000, 
                                   win_length=1024, 
                                   hop_length=256, 
                                   num_mels=80, 
                                   mel_fmin=0, 
                                   mel_fmax=None
    )



# config = FastSpeechConfig(
#     run_name="fast_speech2_ljspeech",
#     audio=audio_config,
#     batch_size=32,
#     eval_batch_size=16,
#     num_loader_workers=8,
#     num_eval_loader_workers=4,
#     compute_input_seq_cache=True,
#     compute_f0=False,
#     run_eval=True,
#     test_delay_epochs=-1,
#     epochs=1000,
#     # text_cleaner="english_cleaners",
# #     energy_cache_path=os.path.join(output_path, "energy_cache"),
#     # use_phonemes=True,
#     # phoneme_language="en-us",
#     # phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
#     precompute_num_workers=8,
#     print_step=50,
#     print_eval=False,
#     mixed_precision=False,
#     max_seq_len=500000,
#     output_path=output_path,
#     characters=vie_char_conf,
#     datasets=[dataset_config],
# )
config = VitsConfig(
        audio=audio_config,
        run_name="vits_viet",
        batch_size=32,
        eval_batch_size=8,
        batch_group_size=5,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        use_phonemes=False,
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False,
        characters=vie_char_conf,
        test_sentences=['xin chào, tất cả mọi người'],
        lr_disc=0.0005,
        lr_gen=0.0005
    )


ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)
model = Vits(config, ap, tokenizer, speaker_manager=None)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter = ljspeech_vi
)

trainer = Trainer(
    TrainerArgs(restore_path = "/kaggle/input/data-vocoder/checkpoint_80000.pth"), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
