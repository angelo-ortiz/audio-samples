import logging
from pathlib import Path

from babblerl.io import read_manifest
from babblerl.constants import ALIGNMENTS_ROOT, AUDIO_ROOT
from babblerl.metrics import _normalize
from babblerl.utils import get_all_transcriptions, read_transcription_and_metrics
from jinja2 import Environment, FileSystemLoader
from pydub import AudioSegment
from sympy import N
from tqdm import tqdm

# Configuration
SPLIT = 'test-2h_snr-40.0_c50-50.0_pesq-3.0_stoi-0.9' # 'test'
TRANSCRIPTION_DIR = 'omniASR_LLM_1B_transcriptions'
CTC_TRANSCRIPTION_DIR = 'omniASR_CTC_3B_transcriptions'
ACOUSTIC_DIR = 'acoustic_transcriptions'
SRC_DATA_DIR = ALIGNMENTS_ROOT.parent / 'resynthesis'
ROOT_DIR = Path('.')
OUTPUT_DIR = ROOT_DIR / 'pages'
DST_DATA_DIR = ROOT_DIR / 'data'
TEMPLATE_DIR = ROOT_DIR / 'templates'
MENU_PAGES = [
    {'src_name': 'pre_trained_sparc/multi', 'dst_name': 'sparc_multi', 'title': 'SPARC-multi', 'filename': 'sparc_multi.html'},
    {'src_name': 'encodec/bw_1-5', 'dst_name': 'encodec_cb2', 'title': 'EnCodec-2', 'filename': 'encodec_2.html'},
    {'src_name': 'encodec/bw_6', 'dst_name': 'encodec_cb8', 'title': 'EnCodec-8', 'filename': 'encodec_8.html'},
    {'src_name': 'hifi_gan/train-50h_snr-35.0_c50-10.0_old-split_s500k', 'dst_name': 'hifi_gan_500k_snr-35_c50-10', 'title': 'HiFi-GAN 500k (no avg. spk)', 'filename': 'hifi_gan.html'},
    {'src_name': 'hifi_gan/train-50h_snr-35.0_c50-10.0_old-split_avg-spk', 'dst_name': 'hifi_gan_500k_snr-35_c50-10_avg-spk', 'title': 'HiFi-GAN 500k', 'filename': 'hifi_gan_avg-spk.html'},
    {'src_name': 'hifi_gan/train-50h_snr-35.0_c50-40.0', 'dst_name': 'hifi_gan_2M_snr-35_c50-40', 'title': 'HiFi-GAN 2M filter+', 'filename': 'hifi_gan_c50-40.html'},
    {'src_name': 'hifi_gan/train-50h_snr-40.0_c50-50.0_pesq-3.0_stoi-0.9_s500k', 'dst_name': 'hifi_gan_2M_snr-40_c50-50_pesq-3_stoi-0-9', 'title': 'HiFi-GAN 2M filter++', 'filename': 'hifi_gan_filtered.html'},
    {'src_name': 'hifi_gan/train-50h_snr-40.0_c50-50.0_pesq-3.0_stoi-0.9_float8-e5m2_s100k', 'dst_name': 'hifi_gan_2M_snr-40_c50-50_pesq-3_stoi-0-9_fp8_800k', 'title': 'HiFi-GAN 800k filter++ float8', 'filename': 'hifi_gan_filtered_fp8.html'},
]
MAX_ROWS_PER_LANGUAGE_BEST = 3 # Limit the number of audio samples displayed per language


def remove_text_within_parentheses(text: str) -> str:
    """Removes the trailing text within parentheses from the input string.

    Args:
        text: The input string.

    Returns:
        The input string with the text within parentheses removed.
    """
    if (beg := text.find('（')) != -1:
        return text[:beg].strip()
    elif (beg := text.find('(')) != -1:
        return text[:beg].strip()
    return text


def format_float_str(value: float | str, precision: int = 2, to_percent: bool = False) -> str:
    """Formats a float value as a string with specified precision.
    
    Args:
        value: The float value to format.
        precision: The number of decimal places.
        to_percent: Whether to convert the value to a percentage.
    """
    if isinstance(value, str):
        return value
    if to_percent:
        value *= 100
    return f"{value:.{precision}f}"


def get_language_data(src_page: str, dst_page: str, lang: str, selected_stems: list[str] | None = None):
    """Generates data for a specific language to be used in the HTML page.

    Args:
        src_page: The source page name.
        dst_page: The destination page name.
        lang: The language code.

    Returns:
        A dictionary containing language data for the HTML page.
    """
    # Read the manifest to get audio file paths
    manifest_path = ALIGNMENTS_ROOT / SPLIT / 'manifests' / f'{lang}.tsv'
    manifest: dict[str, tuple[Path, int]] = read_manifest(manifest_path)

    # Read the transcriptions and metrics
    if not (SRC_DATA_DIR / src_page / SPLIT / TRANSCRIPTION_DIR / f'{lang}.csv').exists():
        return None  # Skip languages without synthesised transcriptions

    gt_trans = read_transcription_and_metrics(ALIGNMENTS_ROOT / SPLIT / TRANSCRIPTION_DIR / f'{lang}.csv')
    syn_trans = read_transcription_and_metrics(SRC_DATA_DIR / src_page / SPLIT / TRANSCRIPTION_DIR / f'{lang}.csv') # TODO: adapt here with another for acoustic metrics

    gt_ctc_trans = read_transcription_and_metrics(ALIGNMENTS_ROOT / SPLIT / CTC_TRANSCRIPTION_DIR / f'{lang}.csv')
    if (ctc_trans_file := SRC_DATA_DIR / src_page / SPLIT / CTC_TRANSCRIPTION_DIR / f'{lang}.csv').exists():
        syn_ctc_trans = read_transcription_and_metrics(ctc_trans_file)
    else:
        syn_ctc_trans = {stem: {} for stem in gt_ctc_trans}

    if selected_stems is None:
        syn_trans_items = sorted(syn_trans.items(), key=lambda x: x[1].get('wer', 0))  # Sort by WER
        selected_stems = [item[0] for item in syn_trans_items[:MAX_ROWS_PER_LANGUAGE_BEST]]
        selected_stems += [item[0] for item in syn_trans_items[-MAX_ROWS_PER_LANGUAGE_BEST:]]

    # Read the original transcriptions
    orig_transcriptions: dict[str, str] = get_all_transcriptions(AUDIO_ROOT / lang, return_header=False)  # type: ignore
    if lang == 'nan-tw':
        orig_transcriptions = {
            stem: _normalize(remove_text_within_parentheses(trans), remove_spaces=False) 
            for stem, trans in orig_transcriptions.items()
        }

    src_syn_audios_dir = SRC_DATA_DIR / src_page / SPLIT / 'audios' / lang
    dst_gt_audios_dir = DST_DATA_DIR / dst_page / lang / 'ground_truth'
    dst_gt_audios_dir.mkdir(parents=True, exist_ok=True)
    dst_syn_audios_dir = DST_DATA_DIR / dst_page / lang / 'synthesised'
    dst_syn_audios_dir.mkdir(parents=True, exist_ok=True)
    gt_audio_paths, syn_audio_paths = [], []
    for stem in selected_stems:
        assert stem in manifest, f"Stem {stem} not found in manifest for language {lang}"
        
        # Convert WAV to MP3 and copy to destination directories
        gt_audio_paths.append(dst_gt_audios_dir / f'{stem}.mp3')
        if not gt_audio_paths[-1].exists():
            _ = AudioSegment.from_wav(manifest[stem][0]).export(gt_audio_paths[-1], format='mp3')
        syn_audio_paths.append(dst_syn_audios_dir / f'{stem}.mp3')
        if not syn_audio_paths[-1].exists():
            _ = AudioSegment.from_wav(src_syn_audios_dir / f'{stem}.wav').export(syn_audio_paths[-1], format='mp3')

    return {
        'name': lang,
        'num_samples': len(selected_stems),
        'ground_truth_audio': [f.relative_to(ROOT_DIR) for f in gt_audio_paths],
        'synthesised_audio': [f.relative_to(ROOT_DIR) for f in syn_audio_paths],
        'original_transcriptions': [orig_transcriptions[stem] for stem in selected_stems],
        'ground_truth_transcriptions': [[trans[stem].get('transcription', '<UNKNOWN>') for stem in selected_stems] for trans in [gt_trans, gt_ctc_trans]],
        'synthesised_transcriptions': [[trans[stem].get('transcription', '<UNKNOWN>') for stem in selected_stems] for trans in [syn_trans, syn_ctc_trans]],
        'cer_ground_truth': [[format_float_str(trans[stem].get('cer', ''), to_percent=True) for stem in selected_stems] for trans in [gt_trans, gt_ctc_trans]],
        'cer_synthesised': [[format_float_str(trans[stem].get('cer', ''), to_percent=True) for stem in selected_stems] for trans in [syn_trans, syn_ctc_trans]],
        'wer_ground_truth': [[format_float_str(trans[stem].get('wer', ''), to_percent=True) for stem in selected_stems] for trans in [gt_trans, gt_ctc_trans]],
        'wer_synthesised': [[format_float_str(trans[stem].get('wer', ''), to_percent=True) for stem in selected_stems] for trans in [syn_trans, syn_ctc_trans]],
        'mcd': [format_float_str(syn_trans[stem].get('mcd', ''), to_percent=False) for stem in selected_stems],
        'f0': [format_float_str(syn_trans[stem].get('f0', ''), to_percent=False) for stem in selected_stems],
    }, selected_stems


if __name__ == '__main__':
    pydub_logger = logging.getLogger("pydub.converter")
    pydub_logger.setLevel(logging.WARNING)

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template('page_template.html')

    lang_selected_stems: dict[str, list[str]] = {}
    for page in tqdm(MENU_PAGES):
        model_dir = Path(SRC_DATA_DIR) / page['src_name'] / SPLIT
        if not model_dir.exists():
            print(f"Warning: Data directory ({model_dir}) for {page['src_name']} does not exist. Skipping.")
            continue
        languages = sorted([d for d in (model_dir / 'audios').iterdir() if d.is_dir()])
        lang_data = [
            get_language_data(page['src_name'], page['dst_name'], lang.name, lang_selected_stems.get(lang.name))
            for lang in languages
        ]
        for lang, data in zip(languages, lang_data):
            if data is not None and lang.name not in lang_selected_stems:
                lang_selected_stems[lang.name] = data[1]
        lang_data = [data[0] for data in lang_data if data is not None]  # Filter out None values
        output_html = template.render(
            page_title=page['title'],
            menu_pages=MENU_PAGES,
            languages=lang_data
        )
        output_path = OUTPUT_DIR / page['filename']
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_html)
