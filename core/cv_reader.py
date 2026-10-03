import os
import glob
import re
import unicodedata
from datetime import datetime
from typing import List, Tuple, Dict, Any, Set, Optional
import joblib
from pathlib import Path
import concurrent.futures
import pymupdf as fitz
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
import docx

# =====================================================================
# TẦNG 1: TIẾP NHẬN VÀ TIỀN XỬ LÝ (INGESTION)
# =====================================================================
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

    # [KHỐI ĐẦU VÀO DỮ LIỆU CHÍNH]
    def run(self, input_path: str) -> List[Tuple[str, str]]:
        raw_data: List[Tuple[str, str]] = []
        
        if os.path.isdir(input_path):
            all_files = glob.glob(os.path.join(input_path, "*"))
            for f in all_files:
                if os.path.isfile(f):
                    raw_data.extend(self.run(f)) 
            return [(cand_id, self.clean_text(raw)) for cand_id, raw in raw_data if raw.strip()]

        ext = os.path.splitext(input_path)[1].lower()

        if ext == ".pdf":
            raw_data.extend(self._read_pdf(input_path))
        elif ext == ".csv":
            raw_data.extend(self._read_csv(input_path))
        elif ext in [".docx", ".doc"]:
            raw_data.extend(self._read_docx(input_path))
        elif ext in [".txt", ".text"]:
            try:
                with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if content.strip():
                    raw_data.extend(self._detect_and_split_records(content, os.path.basename(input_path)))
            except Exception as e:
                print(f"Lỗi đọc file txt {input_path}: {e}")
        else:
            print(f"Bỏ qua file '{os.path.basename(input_path)}': Định dạng '{ext}' chưa được hỗ trợ.")

        return [(cand_id, self.clean_text(raw)) for cand_id, raw in raw_data if raw.strip()]

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
        return [(base_name, full_text.strip())]

    def _read_pdf(self, file_path: str) -> List[Tuple[str, str]]:
        doc = None
        base_name = os.path.basename(file_path)
        try:
            doc = fitz.open(file_path)
            full_text = ""
            for page in doc:
                blocks = page.get_text("blocks")
                text_blocks = [b for b in blocks if b[6] == 0]
                text_blocks.sort(key=lambda b: (b[1], b[0]))
                full_text += "\n".join([b[4] for b in text_blocks]) + "\n"

            if not full_text.strip(): return []
            return self._detect_and_split_records(full_text, base_name)
        except Exception as e:
            print(f"Lỗi đọc pdf {file_path}: {e}")
            return []
        finally:
            if doc: doc.close()

    def _read_docx(self, file_path: str) -> List[Tuple[str, str]]:
        try:
            doc = docx.Document(file_path)
            full_text = "\n".join([para.text for para in doc.paragraphs])
            if not full_text.strip(): return []
            return self._detect_and_split_records(full_text, os.path.basename(file_path))
        except Exception as e:
            print(f"Lỗi đọc docx {file_path}: {e}")
            return []

    def _read_csv(self, csv_path: str) -> List[Tuple[str, str]]:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Lỗi đọc csv {csv_path}: {e}")
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
                results.extend(self._detect_and_split_records(text, base_name))
        return results

# =====================================================================
# TẦNG 2: TRÍCH XUẤT THÔNG TIN (EXTRACTION) - ĐÃ HỢP NHẤT VÀ TỐI ƯU
# =====================================================================
def extract_contact_and_labeled_fields(text: str) -> Dict[str, Any]:
    result = {
        "name": None, "email": None, "phone": None, "github": None,
        "linkedin": None, "age": None, "university": None,
        "raw_position": None, "raw_experience_str": None,
    }
    if not text: return result

    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if email_match: result["email"] = email_match.group(0).rstrip('.')

    # Đã sửa lại Regex bắt số điện thoại tốt hơn cho Việt Nam
    phone_match = re.search(r'(?:\+84|0)(?:[3|5|7|8|9])(?:\d{8}|\d{3}\s\d{2}\s\d{3}|\d{4}\s\d{4})', text)
    if phone_match: result["phone"] = re.sub(r'[^\d+]', '', phone_match.group(0))

    github_match = re.search(r'(?:https?:\/\/)?(?:www\.)?github\.com\/([a-zA-Z0-9_\-]+)', text, re.IGNORECASE)
    if github_match: result["github"] = f"https://github.com/{github_match.group(1)}"

    linkedin_match = re.search(r'(?:https?:\/\/)?(?:www\.)?linkedin\.com\/in\/([a-zA-Z0-9_\-]+)', text, re.IGNORECASE)
    if linkedin_match: result["linkedin"] = f"https://www.linkedin.com/in/{linkedin_match.group(1)}"

    kv_pattern = re.compile(
        r'(?i)(?:^|\n)\s*(Name|Age|University|Position|Experience|Skills|Họ\s*và\s*tên|Tuổi|Trường)\s*[:\-]\s*(.+)'
    )
    labeled_data = {}
    for match in kv_pattern.finditer(text):
        labeled_data[match.group(1).lower().strip()] = match.group(2).strip()

    result["name"] = labeled_data.get("name") or labeled_data.get("họ và tên")
    result["raw_experience_str"] = labeled_data.get("experience")
    return result

class SkillExtractor:
    """Hợp nhất Module load kỹ năng"""
    def __init__(self, skills_path: str = None):
        self.nlp = spacy.blank("en")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        self.taxonomy = {
            "Python": ["python", "py"], "SQL": ["sql", "mysql", "postgresql"],
            "Machine Learning": ["machine learning", "ml", "học máy"],
            "C++": ["c++", "cpp"], "Java": ["java"], "Pandas": ["pandas"]
        }
        
        # Ưu tiên load từ model joblib nếu tồn tại
        if skills_path and Path(skills_path).exists():
            try:
                mlb = joblib.load(skills_path)
                skills = [str(skill) for skill in mlb.classes_]
                patterns = [self.nlp.make_doc(skill) for skill in skills if skill.strip()]
                self.matcher.add("skills", patterns)
                return
            except Exception:
                pass
        
        # Nếu không có file pkl, xài từ điển Taxonomy mặc định
        for canonical_name, aliases in self.taxonomy.items():
            patterns = [self.nlp.make_doc(text) for text in aliases]
            self.matcher.add(canonical_name, patterns)

    def extract_skills(self, text: str) -> List[str]:
        if not text: return []
        results = set()
        doc = self.nlp(text)
        for match_id, start, end in self.matcher(doc):
            string_id = self.nlp.vocab.strings[match_id]
            results.add(string_id) if string_id in self.nlp.vocab.strings else results.add(doc[start:end].text)
        return sorted(list(results))

class ExperienceExtractor:
    def __init__(self):
        self.current_year = datetime.now().year
        self.exp_header = re.compile(r'(?:WORK\s+EXPERIENCE|KINH\s*NGHIỆM)\b', re.IGNORECASE)
        self.next_header = re.compile(r'(?:EDUCATION|HỌC\s*VẤN|SKILLS?|KỸ\s*NĂNG)\b', re.IGNORECASE)

    def _classify_level(self, years: float, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ["senior", "lead"]): return "Senior"
        if any(w in text_lower for w in ["junior"]): return "Junior"
        if any(w in text_lower for w in ["intern", "fresher"]): return "Intern/Fresher"
        
        if years < 1.0: return "Intern/Fresher"
        elif 1.0 <= years < 3.0: return "Junior"
        elif 3.0 <= years < 5.0: return "Mid-Level"
        return "Senior"

    def extract_experience(self, text: str, raw_exp_str: str = None) -> Dict[str, Any]:
        target_text = raw_exp_str or text
        
        # Thuật toán Merge Intervals
        date_ranges = re.findall(r'\b(20\d{2})\s*[-–—tođến]+\s*(20\d{2}|present|nay|hiện\s*tại)\b', target_text, re.IGNORECASE)
        total_years = 0.0
        
        if date_ranges:
            intervals = []
            for start, end in date_ranges:
                start_y = int(start)
                end_y = self.current_year if any(k in end.lower() for k in ["present", "nay", "hiện"]) else int(end)
                if start_y <= end_y: intervals.append([start_y, end_y])
            
            if intervals:
                intervals.sort(key=lambda x: x[0])
                merged = [intervals[0]]
                for current in intervals[1:]:
                    previous = merged[-1]
                    if current[0] <= previous[1]: previous[1] = max(previous[1], current[1])
                    else: merged.append(current)
                
                total_years = float(sum(end - start for start, end in merged))
                return {"years": total_years, "level": self._classify_level(total_years, target_text)}
                
        # Fallback regex số năm
        year_explicit = re.search(r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?|năm)', target_text, re.IGNORECASE)
        if year_explicit:
            yrs = float(year_explicit.group(1))
            return {"years": yrs, "level": self._classify_level(yrs, target_text)}

        return {"years": 0.0, "level": self._classify_level(0.0, target_text)}

# =====================================================================
# TẦNG 3: ĐIỀU PHỐI (ORCHESTRATION) - ĐÃ ĐỔI SANG MULTIPROCESSING
# =====================================================================
class CVParserPipeline:
    def __init__(self):
        self.ingestor = CVDataIngestor()
        self.skill_extractor = SkillExtractor()
        self.exp_extractor = ExperienceExtractor()
        
    def process_single_cv(self, cv_data: Tuple[str, str]) -> dict:
        cv_id, text = cv_data
        contact_info = extract_contact_and_labeled_fields(text)
        skills_list = self.skill_extractor.extract_skills(text)
        experience = self.exp_extractor.extract_experience(text, contact_info.get("raw_experience_str"))
        
        return {
            "candidate_id": cv_id,
            "skills": skills_list,
            "level": experience.get("level"),
            "years": experience.get("years"),
            "email": contact_info.get("email"),
            "phone": contact_info.get("phone")
        }

    def run(self, input_path: str) -> pd.DataFrame:
        print(f"Đang nạp dữ liệu từ: {input_path}...")
        raw_cvs = self.ingestor.run(input_path)
        
        if not raw_cvs:
            print("Không tìm thấy CV nào hợp lệ!")
            return pd.DataFrame()

        print(f"Đã nạp {len(raw_cvs)} CV. Ép xung đa tiến trình (ProcessPool)...")
        final_results = []

        # ĐÃ SỬA: Thay ThreadPool bằng ProcessPool để giải phóng sức mạnh CPU
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.process_single_cv, data) for data in raw_cvs]
            for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                try:
                    final_results.append(future.result())
                except Exception as e:
                    print(f"Lỗi khi xử lý CV: {e}")

                if i % 10 == 0 or i == len(raw_cvs):
                    print(f"  -> Đã bóc tách {i}/{len(raw_cvs)} CV...")

        df_final = pd.DataFrame(final_results)
        print(f"XONG ! Đã return DataFrame với shape: {df_final.shape}")
        return df_final


# =====================================================================
# KHỐI CHẠY KIỂM THỬ THỰC TẾ (RUN SCRIPT)
# =====================================================================
if __name__ == "__main__":
    # ĐỔI ĐƯỜNG DẪN Ở ĐÂY
    INPUT_FOLDER = r"dán đường dẫn"
    OUTPUT_CSV = r"dán đường dẫn"
    
    if os.path.exists(INPUT_FOLDER):
        pipeline = CVParserPipeline()
        df = pipeline.run(INPUT_FOLDER)
        
        if not df.empty:
            print(df.head())
            df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            print(f"Đã lưu kết quả tại: {OUTPUT_CSV}")
    else:
        print(f" Đường dẫn không tồn tại: {INPUT_FOLDER}")