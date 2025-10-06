from __future__ import annotations

import os
import pytest

from marker_pattern.lang import parse_program, compile_program
from utils_llm import llm_retry


requires_groq = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set; real LLM tests are skipped",
)
requires_heavy = pytest.mark.skipif(
    not os.environ.get("RUN_HEAVY_LLM"),
    reason="RUN_HEAVY_LLM not set; heavy LLM tests are skipped",
)


@requires_groq
@requires_heavy
@pytest.mark.llm
@llm_retry(tries=3, delay_s=0.5)
def test_llm_turkey_plate_accuracy() -> None:
    src = (
        "@plate_eval():\n"
        "  ```\n"
        "  reference = [\n"
        "    (\"01\", \"Adana\"),(\"02\", \"Adıyaman\"),(\"03\", \"Afyonkarahisar\"),(\"04\", \"Ağrı\"),(\"05\", \"Amasya\"),\n"
        "    (\"06\", \"Ankara\"),(\"07\", \"Antalya\"),(\"08\", \"Artvin\"),(\"09\", \"Aydın\"),(\"10\", \"Balıkesir\"),\n"
        "    (\"11\", \"Bilecik\"),(\"12\", \"Bingöl\"),(\"13\", \"Bitlis\"),(\"14\", \"Bolu\"),(\"15\", \"Burdur\"),\n"
        "    (\"16\", \"Bursa\"),(\"17\", \"Çanakkale\"),(\"18\", \"Çankırı\"),(\"19\", \"Çorum\"),(\"20\", \"Denizli\"),\n"
        "    (\"21\", \"Diyarbakır\"),(\"22\", \"Edirne\"),(\"23\", \"Elazığ\"),(\"24\", \"Erzincan\"),(\"25\", \"Erzurum\"),\n"
        "    (\"26\", \"Eskişehir\"),(\"27\", \"Gaziantep\"),(\"28\", \"Giresun\"),(\"29\", \"Gümüşhane\"),(\"30\", \"Hakkâri\"),\n"
        "    (\"31\", \"Hatay\"),(\"32\", \"Isparta\"),(\"33\", \"Mersin\"),(\"34\", \"İstanbul\"),(\"35\", \"İzmir\"),\n"
        "    (\"36\", \"Kars\"),(\"37\", \"Kastamonu\"),(\"38\", \"Kayseri\"),(\"39\", \"Kırklareli\"),(\"40\", \"Kırşehir\"),\n"
        "    (\"41\", \"Kocaeli\"),(\"42\", \"Konya\"),(\"43\", \"Kütahya\"),(\"44\", \"Malatya\"),(\"45\", \"Manisa\"),\n"
        "    (\"46\", \"Kahramanmaraş\"),(\"47\", \"Mardin\"),(\"48\", \"Muğla\"),(\"49\", \"Muş\"),(\"50\", \"Nevşehir\"),\n"
        "    (\"51\", \"Niğde\"),(\"52\", \"Ordu\"),(\"53\", \"Rize\"),(\"54\", \"Sakarya\"),(\"55\", \"Samsun\"),\n"
        "    (\"56\", \"Siirt\"),(\"57\", \"Sinop\"),(\"58\", \"Sivas\"),(\"59\", \"Tekirdağ\"),(\"60\", \"Tokat\"),\n"
        "    (\"61\", \"Trabzon\"),(\"62\", \"Tunceli\"),(\"63\", \"Şanlıurfa\"),(\"64\", \"Uşak\"),(\"65\", \"Van\"),\n"
        "    (\"66\", \"Yozgat\"),(\"67\", \"Zonguldak\"),(\"68\", \"Aksaray\"),(\"69\", \"Bayburt\"),(\"70\", \"Karaman\"),\n"
        "    (\"71\", \"Kırıkkale\"),(\"72\", \"Batman\"),(\"73\", \"Şırnak\"),(\"74\", \"Bartın\"),(\"75\", \"Ardahan\"),\n"
        "    (\"76\", \"Iğdır\"),(\"77\", \"Yalova\"),(\"78\", \"Karabük\"),(\"79\", \"Kilis\"),(\"80\", \"Osmaniye\"),\n"
        "    (\"81\", \"Düzce\"),\n"
        "  ]\n"
        "  inv_map = {num: city for (num, city) in reference}\n"
        "  ref_cities = [city for (_, city) in reference]\n"
        "  def norm(s):\n"
        "      s = s.strip().lower()\n"
        "      repl = {\n"
        "        'ş':'s','ı':'i','ï':'i','ğ':'g','ü':'u','ö':'o','ç':'c','â':'a','ê':'e','î':'i','û':'u','ā':'a',\n"
        "        'İ':'i','Ş':'s','Ğ':'g','Ü':'u','Ö':'o','Ç':'c'\n"
        "      }\n"
        "      return ''.join(repl.get(ch, ch) for ch in s)\n"
        "  def parse_pairs(s):\n"
        "      out = []\n"
        "      for tok in s.replace(';', ',').split(','):\n"
        "          tok = tok.strip()\n"
        "          if not tok:\n"
        "              continue\n"
        "          if '-' in tok:\n"
        "              num, name = tok.split('-', 1)\n"
        "          elif ':' in tok:\n"
        "              num, name = tok.split(':', 1)\n"
        "          else:\n"
        "              parts = tok.split()\n"
        "              if parts and parts[0].isdigit():\n"
        "                  num = parts[0]\n"
        "                  name = ' '.join(parts[1:])\n"
        "              else:\n"
        "                  continue\n"
        "          num = num.strip()\n"
        "          if len(num) > 0 and num.isdigit():\n"
        "              num = num.zfill(2)\n"
        "          out.append((num, name.strip()))\n"
        "      return out\n"
        "  def eval_accuracy(cities_str, pairs_str):\n"
        "      cities = [c.strip() for c in cities_str.replace(';', ',').split(',') if c.strip()]\n"
        "      order_correct = sum(1 for i, c in enumerate(cities[:len(ref_cities)]) if norm(c) == norm(ref_cities[i]))\n"
        "      pairs = parse_pairs(pairs_str)\n"
        "      pair_correct = sum(1 for (n, name) in pairs if n in inv_map and norm(name) == norm(inv_map[n]))\n"
        "      total = len(reference)\n"
        "      ratio = (order_correct/total + pair_correct/total) / 2.0\n"
        "      return ratio\n"
        "  ```\n"
        "  Please list all 81 cities of Turkey as a single comma-separated line, in the exact order from 01 to 81: [cities]\n"
        "  Now, using those cities, produce a comma-separated mapping in the form 'NN-City' for all 81 plates in the same order: [pairs]\n"
        "  = <`str(eval_accuracy(cities, pairs))`>\n\n"
        "= <plate_eval()>\n"
    )
    rt = compile_program(parse_program(src))
    out = rt.run_main()
    assert isinstance(out, str)
    try:
        score = float(out.strip())
    except Exception:
        # If the model returns unexpected formatting, consider it a miss
        score = 0.0
    assert 0.0 <= score <= 1.0
    # Expect at least some reasonable accuracy; keep threshold conservative for CI stability
    assert score >= 0.3


