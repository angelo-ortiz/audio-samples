# audio-samples/
# ├── index.html
# ├── templates/
# │   └── page_template.html
# ├── data/
# │   ├── Pre-trained_SPARC-multi/
# │   │   ├── en/
# │   │   │   ├── ground_truth/
# │   │   │   │   ├── sample1.mp3
# │   │   │   │   ├── sample2.mp3
# │   │   │   │   └── ...
# │   │   │   ├── synthesised/
# │   │   │   │   ├── sample1.mp3
# │   │   │   │   ├── sample2.mp3
# │   │   │   │   └── ...
# │   │   │   ├── ground_truth_transcriptions.txt
# │   │   │   ├── synthesised_transcriptions.txt
# │   │   │   ├── metrics.json
# │   │   │   └── ...
# │   │   └── [other languages]/
# │   ├── Pre-trained_SPARC-en+/
# │   │   └── [languages as above]/
# │   ├── Trained_HiFi-GAN/
# │   │   └── [languages as above]/
# ├── assets/
# │   └── style.css
# └── generate_pages.py


import csv
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from babblerl.io import read_manifest
from babblerl.constants import ALIGNMENTS_ROOT
from pydub import AudioSegment

# Configuration
SPLIT = 'test'
SRC_DATA_DIR = ALIGNMENTS_ROOT.parent / 'resynthesis'
DST_DATA_DIR = Path(__file__).parent / 'data'
TEMPLATE_DIR = Path(__file__).parent / 'templates'
OUTPUT_DIR = Path(__file__).parent / 'pages'
MENU_PAGES = [
    {'src_name': 'pre_trained_sparc/multi', 'dst_name': 'sparc_multi', 'title': 'SPARC-multi', 'filename': 'sparc_multi.html'},
    {'src_name': 'pre_trained_sparc/en+', 'dst_name': 'sparc_en_plus', 'title': 'SPARC-en+', 'filename': 'sparc_en_plus.html'},
    {'src_name': 'trained_hifi_gan', 'dst_name': 'hifi_gan', 'title': 'HiFi-GAN', 'filename': 'hifi_gan.html'},
]
MAX_ROWS_PER_LANGUAGE_BEST = 3 # Limit the number of audio samples displayed per language

def load_transcription_wer_cer(path: Path) -> dict[str, tuple[str, float, float]]:
    with open(path, 'r', newline='') as fp:
        reader = csv.reader(fp, delimiter=',')
        header = next(reader)
        assert header == ['file', 'transcription', 'cer', 'wer'], f"Unexpected header in {path}: {header}"
        mapping = {}
        for row in reader:
            stem, transcription, cer, wer = row
            mapping[stem] = (transcription, float(cer), float(wer))
    return mapping

def get_language_data(src_page: str, dst_page: str, lang: str):
    # Read the manifest to get audio file paths
    manifest_path = ALIGNMENTS_ROOT / SPLIT / 'manifests' / f'{lang}.tsv'
    manifest: dict[str, tuple[Path, int]] = read_manifest(manifest_path)

    # Read the transcriptions and metrics
    gt_trans = load_transcription_wer_cer(ALIGNMENTS_ROOT / SPLIT / 'omni_transcriptions' / f'{lang}.csv')
    syn_trans = load_transcription_wer_cer(SRC_DATA_DIR / src_page / SPLIT / 'omni_transcriptions' / f'{lang}.csv')

    syn_trans_items = sorted(syn_trans.items(), key=lambda x: x[1][-1], reverse=True)  # Sort by WER descending
    selected_stems = [item[0] for item in syn_trans_items[:MAX_ROWS_PER_LANGUAGE_BEST]]
    selected_stems += [item[0] for item in syn_trans_items[-MAX_ROWS_PER_LANGUAGE_BEST:]]

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

        # # Write transcriptions to text files
        # for trans, output in zip([gt_trans, syn_trans], ['ground_truth_transcriptions.txt', 'synthesised_transcriptions.txt']):
        #     lines = []
        #     for stem in selected_stems:
        #         lines.append(trans[stem][0])
        #     with open(DST_DATA_DIR / dst_page / lang / output, 'w') as f:
        #         f.write('\n'.join(lines) + '\n')

    return {
        'name': lang,
        'num_samples': len(selected_stems),
        'ground_truth_audio': [f.relative_to(OUTPUT_DIR.parent) for f in gt_audio_paths],
        'synthesised_audio': [f.relative_to(OUTPUT_DIR.parent) for f in syn_audio_paths],
        'ground_truth_transcriptions': [gt_trans[stem][0] for stem in selected_stems],
        'synthesised_transcriptions': [syn_trans[stem][0] for stem in selected_stems],
        'cer_ground_truth': [gt_trans[stem][1] for stem in selected_stems],
        'cer_synthesised': [syn_trans[stem][1] for stem in selected_stems],
        'wer_ground_truth': [gt_trans[stem][2] for stem in selected_stems],
        'wer_synthesised': [syn_trans[stem][2] for stem in selected_stems],
        'mcd': ['N/A' for _ in selected_stems],  # Placeholder, replace with actual MCD if available
    }

def main():
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template('page_template.html')

    for page in MENU_PAGES:
        model_dir = Path(SRC_DATA_DIR) / page['src_name'] / SPLIT
        if not model_dir.exists():
            print(f"Warning: Data directory ({model_dir}) for {page['src_name']} does not exist. Skipping.")
            continue
        languages = [d for d in (model_dir / 'audios').iterdir() if d.is_dir()]
        lang_data = [get_language_data(page['src_name'], page['dst_name'], lang.name) for lang in languages]
        output_html = template.render(
            page_title=page['title'],
            menu_pages=MENU_PAGES,
            languages=lang_data
        )
        output_path = OUTPUT_DIR / page['filename']
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_html)
    # Optionally, generate index.html as a redirect or landing page

if __name__ == '__main__':
    main()