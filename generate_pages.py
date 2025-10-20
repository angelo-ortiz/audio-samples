import csv
from pathlib import Path

from babblerl.io import read_manifest
from babblerl.constants import ALIGNMENTS_ROOT, AUDIO_ROOT
from babblerl.utils import get_all_transcriptions
from jinja2 import Environment, FileSystemLoader
from pydub import AudioSegment

# Configuration
SPLIT = 'test'
TRANSCRIPTION_DIR = 'omni_transcriptions_wo_f0'
OUTPUT_DIR = Path('.')
SRC_DATA_DIR = ALIGNMENTS_ROOT.parent / 'resynthesis'
DST_DATA_DIR = Path('data')
TEMPLATE_DIR = Path('templates')
MENU_PAGES = [
    {'src_name': 'pre_trained_sparc/multi', 'dst_name': 'sparc_multi', 'title': 'SPARC-multi', 'filename': 'sparc_multi.html'},
    {'src_name': 'pre_trained_sparc/en+', 'dst_name': 'sparc_en_plus', 'title': 'SPARC-en+', 'filename': 'sparc_en_plus.html'},
    {'src_name': 'trained_hifi_gan', 'dst_name': 'hifi_gan', 'title': 'HiFi-GAN', 'filename': 'hifi_gan.html'},
]
MAX_ROWS_PER_LANGUAGE_BEST = 3 # Limit the number of audio samples displayed per language


def remove_text_within_parentheses(text: str) -> str:
    """Removes the trailing text within parentheses from the input string.

    Args:
        text: The input string.

    Returns:
        The input string with the text within parentheses removed.
    """
    if (beg := text.find('(')) != -1 and text.find(')') > beg:
        return text[:beg].strip()
    else:
        return text


def load_transcription_and_metrics(path: Path) -> dict[str, dict[str, str | float]]:
    """Loads transcription and metrics from a CSV file.

    Args:
        path: The path to the CSV file.

    Returns:
        A dictionary mapping file names to their transcription and metric data.
    """
    with open(path, 'r', newline='') as fp:
        reader = csv.reader(fp, delimiter=',')
        header = next(reader)
        field2idx = {metric: i for i, metric in enumerate(header)}
        mapping: dict[str, dict[str, str | float]] = {}
        for row in reader:
            mapping[row[field2idx['file']]] = {
                field: row[idx] if field == 'transcription' else float(row[idx]) for field, idx in field2idx.items() if field != 'file'
            }
    return mapping

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

def get_language_data(src_page: str, dst_page: str, lang: str) -> dict[str, str | int | list[str]] | None:
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

    gt_trans = load_transcription_and_metrics(ALIGNMENTS_ROOT / SPLIT / TRANSCRIPTION_DIR / f'{lang}.csv', include_mcd=False)
    syn_trans = load_transcription_and_metrics(SRC_DATA_DIR / src_page / SPLIT / TRANSCRIPTION_DIR / f'{lang}.csv', include_mcd=True)

    syn_trans_items = sorted(syn_trans.items(), key=lambda x: x[1].get('wer', 0))  # Sort by WER
    selected_stems = [item[0] for item in syn_trans_items[:MAX_ROWS_PER_LANGUAGE_BEST]]
    selected_stems += [item[0] for item in syn_trans_items[-MAX_ROWS_PER_LANGUAGE_BEST:]]

    # Read the original transcriptions
    orig_transcriptions: dict[str, str] = get_all_transcriptions(AUDIO_ROOT / lang, return_header=False)  # type: ignore
    if lang == 'nan-tw':
        orig_transcriptions = {stem: remove_text_within_parentheses(trans) for stem, trans in orig_transcriptions.items()}

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
        AudioSegment.from_wav(manifest[stem][0]).export(gt_audio_paths[-1], format='mp3')
        syn_audio_paths.append(dst_syn_audios_dir / f'{stem}.mp3')
        AudioSegment.from_wav(src_syn_audios_dir / f'{stem}.wav').export(syn_audio_paths[-1], format='mp3')

    return {
        'name': lang,
        'num_samples': len(selected_stems),
        'ground_truth_audio': [f.relative_to(OUTPUT_DIR) for f in gt_audio_paths],
        'synthesised_audio': [f.relative_to(OUTPUT_DIR) for f in syn_audio_paths],
        'original_transcriptions': [orig_transcriptions[stem] for stem in selected_stems],
        'ground_truth_transcriptions': [gt_trans[stem].get('transcription', '') for stem in selected_stems],
        'synthesised_transcriptions': [syn_trans[stem].get('transcription', '') for stem in selected_stems],
        'cer_ground_truth': [format_float_str(gt_trans[stem].get('cer', ''), to_percent=True) for stem in selected_stems],
        'cer_synthesised': [format_float_str(syn_trans[stem].get('cer', ''), to_percent=True) for stem in selected_stems],
        'wer_ground_truth': [format_float_str(gt_trans[stem].get('wer', ''), to_percent=True) for stem in selected_stems],
        'wer_synthesised': [format_float_str(syn_trans[stem].get('wer', ''), to_percent=True) for stem in selected_stems],
        'mcd': [format_float_str(syn_trans[stem].get('mcd', ''), to_percent=False) for stem in selected_stems],
        'f0': [format_float_str(syn_trans[stem].get('f0', ''), to_percent=False) for stem in selected_stems],
    }

def main():
    """Main function to generate HTML pages for audio samples."""
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template('page_template.html')

    for page in MENU_PAGES:
        model_dir = Path(SRC_DATA_DIR) / page['src_name'] / SPLIT
        if not model_dir.exists():
            print(f"Warning: Data directory ({model_dir}) for {page['src_name']} does not exist. Skipping.")
            continue
        languages = sorted([d for d in (model_dir / 'audios').iterdir() if d.is_dir()])
        lang_data = [get_language_data(page['src_name'], page['dst_name'], lang.name) for lang in languages]
        lang_data = [data for data in lang_data if data is not None]  # Filter out None values
        output_html = template.render(
            page_title=page['title'],
            menu_pages=MENU_PAGES,
            languages=lang_data
        )
        output_path = OUTPUT_DIR / page['filename']
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_html)

if __name__ == '__main__':
    main()