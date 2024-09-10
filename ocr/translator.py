import transformers.utils
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def load_model():
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX", cache_dir="data/translator")

    
    return model, tokenizer

def translate(Input_text, Input_lang):
    languages = {'Arabic' : 'ar_AR', 'Czech' : 'cs_CZ', 'German' : 'de_DE', 'English' : 'en_XX', 'Spanish' : 'es_XX', 'Estonian' :  'et_EE',
             'Finnish' : 'fi_FI', 'French' : 'fr_XX', 'Gujarati' : 'gu_IN', 'Hindi' : 'hi_IN', 'Italian' : 'it_IT', 'Japanese' : 'ja_XX',
             'Kazakh' : 'kk_KZ', 'Korean' : 'ko_KR', 'Lithuanian' : 'lt_LT', 'Latvian' : 'lv_LV', 'Burmese' : 'my_MM', 'Nepali' : 'ne_NP',
             'Dutch' : 'nl_XX', 'Romanian' : 'ro_RO', 'Russian' : 'ru_RU', 'Sinhala' : 'si_LK', 'Turkish' : 'tr_TR', 'Vietnamese' : 'vi_VN',
             'Chinese' : 'zh_CN', 'Afrikaans' : 'af_ZA', 'Azerbaijani' : 'az_AZ', 'Bengali' : 'bn_IN', 'Persian' : 'fa_IR', 
             'Hebrew' : 'he_IL', 'Croatian' : 'hr_HR', 'Indonesian' : 'd_ID', 'Georgian' : 'ka_GE', 'Khmer' : 'km_KH', 'Macedonian' : 'mk_MK',              'Malayalam' : 'ml_IN', 'Mongolian' : 'mn_MN', 'Marathi' : 'mr_IN', 'Polish' : 'pl_PL', 'Pashto' : 'ps_AF',
             'Portuguese' : 'pt_XX', 'Swedish' : 'sv_SE', 'Swahili' : 'sw_KE', 'Tamil' : 'ta_IN', 'Telugu' : 'te_IN', 'Thai' : 'th_TH',                    'Tagalog' : 'tl_XX', 'Ukrainian' : 'uk_UA', 'Urdu' : 'ur_PK', 'Xhosa' : 'xh_ZA', 'Galician' : 'gl_ES', 'Slovene' : 'sl_SI'}
    model, tokenizer = load_model()
    model_inputs = tokenizer(Input_text, return_tensors="pt")
    generated_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.lang_code_to_id[languages[Input_lang]])
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    return translation[0]
