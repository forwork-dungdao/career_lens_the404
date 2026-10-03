import os
import glob
import re
import unicodedata
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import concurrent.futures
import fitz  # PyMuPDF
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
import json
import docx
class CVDataIngestor:
    def __init__(self, text_col: str = None, id_col: str = None):
        self.text_col = text_col
        self.id_col = id_col

        self.anchor_prefixes = [
            r'cv', r'resume', r'candidate', r'applicant', r'profile', r'person',
            r'ứng\s*viên', r'hồ\s*sơ', r'thí\s*sinh'
        ]
        prefix_group = "|".join(self.anchor_prefixes)
        self.numbered_anchor_pattern = re.compile(
            rf'(?:^|\n)\s*(?:(?P<full_anchor>(?:{prefix_group})\s*(?:#|no\.?|số|-|:)?\s*\d+[:\.\-]?))',
            re.IGNORECASE
        )

        self.name_field_pattern = re.compile(
            r'(?:^|\n)\s*(?:(?P<name_anchor>(?:name|họ\s*và\s*tên|full\s*name)\s*[:\-]))',
            re.IGNORECASE
        )
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')

    def clean_text(self, raw_text: str) -> str:
        if not raw_text or not isinstance(raw_text, str): return ""
        text = unicodedata.normalize("NFC", raw_text)
        text = text.replace('\xa0', ' ').replace('\u200b', ' ')
        text = re.sub(r'[•▪►◆★\*\-–—]+', '-', text)
        lines = [line.strip() for line in text.splitlines()]
        cleaned_lines = []
        for line in lines:
            if line:
                line = re.sub(r'[ \t]+', ' ', line)
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
        return "\n".join(cleaned_lines)

    def _detect_and_split_records(self, full_text: str, base_name: str) -> List[Tuple[str, str]]:
        matches = list(self.numbered_anchor_pattern.finditer(full_text))
        if len(matches) >= 2:
            parts = []
            for i in range(len(matches)):
                start_idx = matches[i].start()
                end_idx = matches[i+1].start() if i + 1 < len(matches) else len(full_text)
                chunk = full_text[start_idx:end_idx].strip()
                anchor_name = matches[i].group("full_anchor").strip()
                cand_id = re.sub(r'[\s#:\.\-]+', '_', anchor_name)
                parts.append((cand_id, chunk))
            return parts

        name_matches = list(self.name_field_pattern.finditer(full_text))
        if len(name_matches) >= 2:
            parts = []
            for i in range(len(name_matches)):
                start_idx = name_matches[i].start()
                end_idx = name_matches[i+1].start() if i + 1 < len(name_matches) else len(full_text)
                chunk = full_text[start_idx:end_idx].strip()
                cand_id = f"{base_name}_candidate_{i+1}"
                parts.append((cand_id, chunk))
            return parts

        email_matches = list(self.email_pattern.finditer(full_text))
        unique_emails = []
        seen = set()
        for m in email_matches:
            if m.group(0) not in seen:
                unique_emails.append(m)
                seen.add(m.group(0))

        if len(unique_emails) >= 2:
            parts = []
            for i in range(len(unique_emails)):
                start_idx = 0
                if i > 0:
                    search_area = full_text[:unique_emails[i].start()]
                    last_double_newline = search_area.rfind('\n\n')
                    start_idx = last_double_newline if last_double_newline != -1 else unique_emails[i].start() - 50

                end_idx = len(full_text)
                if i + 1 < len(unique_emails):
                    next_search_area = full_text[:unique_emails[i+1].start()]
                    next_double_newline = next_search_area.rfind('\n\n')
                    end_idx = next_double_newline if next_double_newline != -1 else unique_emails[i+1].start() - 50

                chunk = full_text[start_idx:end_idx].strip()
                cand_id = f"{base_name}_email_{i+1}"
                parts.append((cand_id, chunk))
            return parts

        return [(base_name, full_text.strip())]

    def _read_pdf(self, file_path: str) -> List[Tuple[str, str]]:
        doc = None
        base_name = os.path.basename(file_path)
        try:
            doc = fitz.open(file_path)
            full_text = ""
            total_chars = 0

            for page in doc:
                blocks = page.get_text("blocks")
                text_blocks = [b for b in blocks if b[6] == 0]
                text_blocks.sort(key=lambda b: (b[1], b[0]))
                page_text = "\n".join([b[4] for b in text_blocks])
                full_text += page_text + "\n"
                total_chars += len(page_text.strip())

            if len(doc) > 0 and (total_chars / len(doc)) < 50:
                print(f"⚠️️ CẢNH BÁO: '{base_name}' có mật độ chữ cực thấp. Khả năng cao là PDF ảnh (Scanned PDF).")

            if not full_text.strip(): return []
            return self._detect_and_split_records(full_text, base_name)
        except Exception as e:
            print(f"⚠ lỗi đọc pdf {file_path}: {e}")
            return []
        finally:
            if doc: doc.close()
    # Đọc file docx 
    def _read_docx(self, file_path: str) -> List[Tuple[str, str]]:
        try:
            doc = docx.Document(file_path)
            full_text = "\n".join([para.text for para in doc.paragraphs])
            if not full_text.strip(): return []
            return self._detect_and_split_records(full_text, os.path.basename(file_path))
        except Exception as e:
            print(f"⚠️ lỗi đọc docx {file_path}: {e}")
            return []

    # CSV cũng được quét qua bộ cắt CV (nếu 1 ô Excel chứa nhiều CV gộp)
    def _read_csv(self, csv_path: str) -> List[Tuple[str, str]]:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"⚠️️ lỗi đọc csv {csv_path}: {e}")
            return []
        col = self.text_col
        if not col or col not in df.columns:
            candidates = [c for c in df.columns if any(k in c.lower() for k in ["resume", "cv", "text", "content"])]
            col = candidates[0] if candidates else df.columns[0]
        
        texts = df[col].fillna("").astype(str).tolist()
        results = []
        for i, text in enumerate(texts):
            if text.strip():
                base_name = f"{os.path.basename(csv_path)}_row_{i}"
                # Áp dụng xé lẻ CV ngay trong ô CSV
                results.extend(self._detect_and_split_records(text, base_name))
        return results

    def run(self, input_path: str) -> List[Tuple[str, str]]:
        raw_data: List[Tuple[str, str]] = []
        if os.path.isdir(input_path):
            all_files = glob.glob(os.path.join(input_path, "*"))
            for f in all_files: raw_data.extend(self.run(f))
            return raw_data
        
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".pdf": raw_data.extend(self._read_pdf(input_path))
        elif ext == ".csv": raw_data.extend(self._read_csv(input_path))
        elif ext in [".docx", ".doc"]: raw_data.extend(self._read_docx(input_path)) # Kích hoạt docx
        elif ext in [".txt", ".text"]:
            try:
                with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if content.strip():
                    raw_data.extend(self._detect_and_split_records(content, os.path.basename(input_path)))
            except Exception as e:
                print(f"⚠️ lỗi txt: {e}")
        return [(cand_id, self.clean_text(raw)) for cand_id, raw in raw_data if raw.strip()]


