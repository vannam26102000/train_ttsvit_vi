from TTS.api import TTS



tts = TTS('cc',
          model_path='/home/nam/Downloads/weight_tts(endtoend)/checkpoint_80000.pth',
          config_path='/home/nam/Downloads/weight_tts(endtoend)/config.json')
        #   vocoder_path='/home/nam/Downloads/Vocoder/checkpoint_50000.pth',
        #   vocoder_config_path='/home/nam/Downloads/Vocoder/config.json')
# Run TTS
# ❗ Since this model is multi-speaker and multi-lingual, we must set the target speaker and the language
# Text to speech with a numpy output
# Text to speech to a file
tts.tts_to_file(text="tôi là công nghệ lõi được phát triển bởi chu văn nam, xịn xò và có đầy đủ tài liệu cũng như phương pháp", file_path="/home/nam/code_python/infer_tts(mel_vocoder)/output11.wav")