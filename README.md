# Imagines Nummorum VLM Indexcard Data Extraction Pipeline

## 🔍 Überblick

> Ein Computer Vision-Skript zur automatisierten Analyse von Karteikarten einer Sammlung von Münzfälschungen mit dem Qwen2.5-VL Vision-Language-Model.

**Aktuelle Version:** 1.0  
**Datum:** August 2025

### Hauptfunktionen

- 🤖 **Mehrstufige KI-Analyse** mit dem Qwen2.5-VL Vision-Language-Model
- 📝 **Automatische Inhaltsklassifikation** (Formulare, Textseiten, leere Seiten)
- 🔍 **Intelligente Münzerkennung** mit Bounding-Box-Extraktion
- 📊 **Strukturierte Metadatenextraktion** von Katalogkarten
- 🔤 **Hybrid-OCR-Verarbeitung** (Tesseract + KI-basierte Texterkennung)
- 🖼️ **Automatisiertes Bildzuschneiden** mit intelligenter Randerkennung
- 📁 **Stapelverarbeitung** mit Fortschrittsanzeige
- 📈 **CSV-Export** für Datenbankintegration und Analyse

## A. Allgemeine Informationen

**Titel der Software:**  
Imagines Nummorum VLM Data Extraction Pipeline

**Kurze Beschreibung der Software:**  
Ein Tool zur automatisierten Analyse von Karteikarten mittels Vision-Language-Model (Qwen2.5-VL). Das System führt eine mehrstufige Bildanalyse durch, klassifiziert Bilder, erkennt handschriftlichen Inhalt und extrahiert strukturierte Daten.

**Software-Typ:**  
Computer Vision Pipeline / Datenverarbeitungs-Skript

**Programmiersprache(n) und Format(e):**  
Python, JSON, hOCR XML, CSV

**Titel des Forschungsprojekts:**  
Zuarbeit für das Vorhaben Imagines Nummorum (https://www.imagines-nummorum.eu/de)

**Kurze Beschreibung des Forschungsprojekts und seiner Ziele:**  
Das Projekt Imagines Nummorum digitalisiert und erschließt antike Münzen. Diese Software unterstützt die automatisierte Extraktion und Strukturierung von Daten aus digitalisierten Karteikarten zur Erstellung einer Forschungsdatenbank.


**Angaben zu Autor:in, Herausgeber:in und Entwickler:in**

- **Name:** Tim Westphal
- **Einrichtung:** Berlin-Brandenburgische Akademie der Wissenschaften - TELOTA
- **Adresse:** Jägerstraße 22/23, 10117 Berlin
- **E-Mail:** tim.westphal@bbaw.de
- **ORCID:** https://orcid.org/0009-0000-8580-4558

**Datum der Softwareentwicklung:**  
2025-06

**Sprache der Software und Dokumentation:**  
deu (Deutsch), eng (Englisch)

**Kompatibilität mit Plattformen:**  
Windows, MacOS, Linux

## 🚀 Schnellstart

### Voraussetzungen

- Python 3.8+ (3.9+ empfohlen)
- 16GB+ RAM (64GB+ für optimale Leistung)
- GPU mit 8GB+ VRAM (optional aber empfohlen)

### Installation

```bash
git clone https://github.com/telota/imagines-nummorum-vlm-data-extraction.git
cd imagines-nummorum-vlm-data-extraction
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Grundlegende Nutzung

```bash
python src/coin_card_information_extraction.py
```

Für detaillierte Installationsanweisungen siehe [INSTALLATION.md](docs/INSTALLATION.md).

## 📚 Dokumentation

| Dokument                                                    | Beschreibung                           |
| ----------------------------------------------------------- | -------------------------------------- |
| [Installationsanleitung](docs/INSTALLATION.md)              | Vollständige Installationsanweisungen  |
| [Benutzerhandbuch](docs/USER_GUIDE.md)                      | Schritt-für-Schritt Anwendungstutorial |
| [Technische Dokumentation](docs/TECHNICAL_DOCUMENTATION.md) | Systemarchitektur und Algorithmen      |
| [API-Referenz](docs/API_REFERENCE.md)                       | Funktions- und Klassendokumentation    |
| [Fehlerbehebung](docs/TROUBLESHOOTING.md)                   | Häufige Probleme und Lösungen          |

## B. Software-Übersicht & Dateistruktur

**Funktionen der Software:**

1. **Mehrstufige Bildanalyse**:

   - Klassifikation als "form", "text_page" oder "empty_page"
   - Handschriftenerkennung
   - Bedingte Inhaltsextraktion basierend auf Bildtyp

2. **OCR-Verarbeitung**:

   - Tesseract OCR für hOCR XML-Format
   - Qwen-VL für KI-basierte Texterkennung
   - Konfigurierbare Strategien (fallback, parallel, etc.)

3. **Strukturierte Datenextraktion**:

   - Münzmetadaten von Katalogkarten
   - Bounding Box-Erkennung für Münzen
   - Automatisches Ausschneiden von Münzbildern

4. **Batch-Verarbeitung**:
   - Rekursive Verzeichnisverarbeitung
   - Fortschrittsanzeige mit tqdm
   - Fehlerbehandlung und Wiederholungsversuche

**Dateiliste/Struktur des Repositoriums:**

```
├── src/
│   ├── coin_card_information_extraction.py  # Hauptpipeline
│   ├── json_to_csv.py                      # CSV-Export-Tool
│   └── validate_data.py                    # Validierungstool
├── data/
│   ├── example_input/                      # Beispielbilder
│   └── example_output/                     # Beispielergebnisse
├── docs/                                   # Dokumentation
├── requirements.txt                        # Python-Abhängigkeiten
├── README.md                              # Diese Datei
├── docs/
│   ├── INSTALLATION.md                    # Installationsanleitung
│   ├── USER_GUIDE.md                     # Benutzerhandbuch
│   ├── TECHNICAL_DOCUMENTATION.md        # Technische Dokumentation
│   ├── API_REFERENCE.md                  # API-Referenz
│   └── TROUBLESHOOTING.md                # Fehlerbehebung
└── TROUBLESHOOTING.md                    # Fehlerbehebung
```

**Gibt es mehrere Versionen der Software?**

- Aktuell: Version 1.0 (August 2025)
- Zukünftige Versionen werden mit Release-Notes über GitHub veröffentlicht
- Zenodo-Integration für DOI-Vergabe geplant

**Abhängigkeiten und Installationsanweisungen:**

### Systemvoraussetzungen:

- **Python**: 3.8 oder höher (3.9+ empfohlen)
- **RAM**: 16GB minimum, 64GB+ für optimale Performance
- **Speicher**: 100GB+ für Model-Cache
- **GPU**: NVIDIA GPU mit 8GB+ VRAM (optional)
- **Tesseract OCR**: Version 4.0+ (optional für OCR-Funktionalität)

### Python-Abhängigkeiten:

```
torch>=2.0.0
torchvision
torchaudio
transformers>=4.35.0
accelerate
Pillow>=9.0.0
requests
jsonschema
pytesseract
tqdm
hf-xet
qwen_vl_utils
natsort
```

Detaillierte Installationsanweisungen finden Sie in [INSTALLATION.md](docs/INSTALLATION.md).

## C. Gemeinsame Nutzung/Zugang zu Informationen

**Wurde die Software von einer anderen Quelle abgeleitet?**

- Teilweise - basiert auf Hugging Face Transformers und Qwen2.5-VL Model
  - Qwen2.5-VL: https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct
  - Transformers Library: https://github.com/huggingface/transformers
  - Tesseract OCR: https://github.com/tesseract-ocr/tesseract

**Für die Software geltende Lizenzen/Einschränkungen:**  
MIT License: https://opensource.org/licenses/MIT

**Links zu Veröffentlichungen, in denen die Software zitiert oder verwendet wird:**  
[Wird bei Veröffentlichung ergänzt]

**Links zu anderen öffentlich zugänglichen Stellen der Software:**

- GitHub Repository: https://github.com/telota/imagines-nummorum-vlm-data-extraction
- Zenodo DOI: [Bei Release verfügbar]

**Links/Beziehungen zu ergänzenden Datensätzen oder Tools:**

- Qwen2.5-VL Model: https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- Imagines Nummorum Project: https://www.imagines-nummorum.eu/de

**Zitation:**  
Imagines Nummorum VLM indexcard data extraction script. Entwickelt von Tim Westphal (2025). Berlin-Brandenburgische Akademie der Wissenschaften - TELOTA. URL: https://github.com/telota/imagines-nummorum-vlm-data-extraction; DOI: [Zenodo DOI bei Release].

## D. Informationen zur Methodik und Entwicklung

**Beschreibung der Methoden und Software-Tools zur Entwicklung:**

- **Vision-Language-Model (VLM)**: Qwen2.5-VL für multimodale Bildanalyse
- **Deep Learning Framework**: PyTorch mit CUDA-Unterstützung
- **Transformers**: Hugging Face Transformers-Bibliothek
- **OCR-Integration**: Tesseract OCR mit pytesseract-Wrapper
- **Bildverarbeitung**: Pillow (PIL) für Bildmanipulation

**Verwendete Methoden zur Datenverarbeitung oder Analyse:**

1. **Mehrstufige Pipeline-Architektur**:

   - Stufe 1: Bildklassifikation und Handschriftenerkennung
   - Stufe 2: Bedingte Inhaltsextraktion basierend auf Bildtyp

2. **Hybrid-OCR-Ansatz**:

   - Tesseract für strukturierte hOCR-Ausgabe
   - VLM für kontextuelle Texterkennung
   - Konfigurierbare Strategien (parallel, fallback)

3. **Intelligente Münzbildextraktion**:

   - Bounding Box-Erkennung mit VLM
   - Adaptive Margin-Erkennung für optimale Ausschnitte
   - Kantenanalyse für Hintergrund-Erkennung

4. **JSON-Schema-Validierung**:
   - Strukturierte Datenextraktion mit Schema-Validierung
   - Retry-Mechanismus für robuste Verarbeitung
   - Fehlerbehandlung und Logging

**Geräte- und/oder softwarespezifische Anforderungen:**

- **Betriebssystem**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: Version 3.8+ mit pip und venv
- **GPU**: NVIDIA GPU mit CUDA 11.8+ (empfohlen)
- **RAM**: 16GB minimum, 64GB+ für 32B-Modell
- **Speicher**: 100GB+ für Modell-Cache und Verarbeitung
- **Netzwerk**: Internetverbindung für initiale Modell-Downloads

**Qualitätssicherungsverfahren:**

- **Automatisierte Validierung**: JSON-Schema-Validierung für alle Ausgaben
- **Retry-Mechanismus**: Bis zu 3 Wiederholungsversuche bei API-Fehlern
- **Progressives Logging**: Detaillierte Logs für Debugging und Monitoring
- **Batch-Validierung**: Vollständigkeitsprüfung mit `validate_data.py`
- **Fallback-Strategien**: OCR-Fallback-Mechanismen für Robustheit

**Angaben zu Standards:**

- **JSON**: Strukturierte Datenausgabe nach JSON-Schema
- **hOCR**: Standard-XML-Format für OCR-Ergebnisse
- **CSV**: Kompatible Ausgabe für Datenbanken und Analysetools
- **Hugging Face Standards**: Modell-Integration über standardisierte APIs

## E. Software-spezifische Informationen

### Hauptmodul: coin_card_information_extraction.py

**Funktionsübersicht:**

- `load_model_and_processor()`: Lädt Qwen2.5-VL Modell
- `process_single_image_multi_stage()`: Verarbeitet einzelnes Bild
- `batch_process_images_multi_stage()`: Batch-Verarbeitung
- `_call_qwen_vl_with_retry()`: Robuste VLM-API-Aufrufe
- `_find_optimal_margin()`: Intelligente Margin-Erkennung für Münzausschnitte

**Parameterliste:**

- `INPUT_IMAGE_DIRECTORY` (String): Eingabeverzeichnis für Bilder
- `OUTPUT_JSON_DIRECTORY` (String): Ausgabeverzeichnis für Ergebnisse
- `MODEL_ID` (String): Hugging Face Modell-ID (default: "Qwen/Qwen2.5-VL-32B-Instruct")
- `OCR_STRATEGY_FOR_TEXT_PAGES` (String): OCR-Strategie ("both", "tesseract_hocr_only", "qwen_text_only")
- `MAX_JSON_RETRIES` (Integer): Maximale Wiederholungsversuche (default: 3)
- `CROP_MARGIN_PIXELS` (Integer): Standard-Margin für Münzausschnitte (default: 40)

**Verwendete Datenformate:**

- **Input**: JPG, PNG, TIFF, BMP Bilddateien
- **Output**: JSON-Dateien, hOCR XML, PNG-Ausschnitte

### Datenkonvertierung: json_to_csv.py

**Funktionsübersicht:**

- `find_json_files()`: Rekursive JSON-Datei-Suche mit natürlicher Sortierung
- `parse_json_file()`: Extrahiert Daten für CSV-Export
- `natural_sort_key()`: Sortierschlüssel für alphanumerische Sortierung

**Output-Spalten:**

- Grunddaten: `file_name`, `image_type`, `status`
- Münzdaten: `num_coins`, `coin1_description`, `coin1_bbox_*`
- Kartendaten: `card_Atelier`, `card_Date`, `card_Métal`, etc.

### Validierung: validate_data.py

**Funktionsübersicht:**

- `verify_processing_status()`: Überprüft Vollständigkeit der Verarbeitung
- `print_results()`: Formatierte Ausgabe der Validierungsergebnisse

**Fehlercodes/Status:**

- `"success"`: Erfolgreiche Verarbeitung
- `"classification_failed"`: Klassifikationsfehler
- `"form_extraction_failed"`: Formular-Extraktionsfehler
- `"text_extraction_failed"`: Text-Extraktionsfehler

## 🔧 Erweiterte Konfiguration

Für detaillierte Konfigurationsoptionen siehe:

- [Technische Dokumentation](docs/TECHNICAL_DOCUMENTATION.md) - Technische Dokumentation
- [API-Referenz](docs/API_REFERENCE.md) - Funktions-Parameter
- [Benutzerhandbuch](docs/USER_GUIDE.md) - Praxisbeispiele

## 🤝 Beitrag und Entwicklung

Dieses Projekt ist Teil des Imagines Nummorum-Vorhabens der Berlin-Brandenburgischen Akademie der Wissenschaften. Beiträge und Verbesserungsvorschläge sind willkommen:

- GitHub Issues für Fehlerberichte
- Pull Requests für Code-Beiträge
- Diskussionen für Funktionswünsche

## 📞 Kontakt und Support

**Entwickler:** Tim Westphal  
**Institution:** Berlin-Brandenburgische Akademie der Wissenschaften - TELOTA  
**E-Mail:** tim.westphal@bbaw.de  
**ORCID:** https://orcid.org/0009-0000-8580-4558

Für technischen Support siehe [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) oder kontaktieren Sie uns direkt.

---

_Diese Software wurde im Rahmen des Imagines Nummorum-Projekts entwickelt und steht unter MIT-Lizenz zur freien Verfügung._
