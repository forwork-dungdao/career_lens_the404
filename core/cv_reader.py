import os
import glob
import re
import unicodedata
from datetime import datetime
from typing import List, Tuple, Dict, Any, Set, Optional
import concurrent.futures
import fitz  
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
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
                print(f" CẢNH BÁO: '{base_name}' có mật độ chữ cực thấp. Khả năng cao là PDF ảnh (Scanned PDF).")

            if not full_text.strip(): return []
            return self._detect_and_split_records(full_text, base_name)
        except Exception as e:
            print(f"lỗi đọc pdf {file_path}: {e}")
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
            print(f"lỗi đọc docx {file_path}: {e}")
            return []

    # CSV cũng được quét qua bộ cắt CV (nếu 1 ô Excel chứa nhiều CV gộp)
    def _read_csv(self, csv_path: str) -> List[Tuple[str, str]]:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"lỗi đọc csv {csv_path}: {e}")
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

#Module 1 : Thiết kế regex

def extract_contact_and_labeled_fields(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "name": None, "email": None, "phone": None, "github": None,
        "linkedin": None, "age": None, "university": None,
        "raw_position": None, "raw_experience_str": None,
    }
    if not text: return result

    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if email_match: result["email"] = email_match.group(0).rstrip('.')

    # Regex bao quát SĐT quốc tế (Hỗ trợ +, dấu cách, dấu ngoặc)
    phone_match = re.search(r'(?:\+?\d{1,4}[\s\.-]?)?(?:\(?\d{2,4}\)?[\s\.-]?)?\d{3,4}[\s\.-]?\d{3,4}', text)
    if phone_match: 
        # Lọc sạch chỉ để lại số và dấu +
        clean_phone = re.sub(r'[^\d+]', '', phone_match.group(0))
        # Ràng buộc độ dài hợp lý của SĐT (9-15 số)
        if 9 <= len(clean_phone) <= 15:
            result["phone"] = clean_phone

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

    raw_age = labeled_data.get("age") or labeled_data.get("tuổi")
    if raw_age:
        age_num = re.search(r'\d+', raw_age)
        if age_num: result["age"] = int(age_num.group(0))

    result["university"] = labeled_data.get("university") or labeled_data.get("trường")
    result["raw_position"] = labeled_data.get("position")
    result["raw_experience_str"] = labeled_data.get("experience")

    if not result["name"]:
        non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        blacklist_words = ["cv", "resume", "curriculum", "intern", "engineer", "developer", "page"]
        for line in non_empty_lines[:3]:
            words = line.split()
            if 2 <= len(words) <= 5 and not re.search(r'[@\d/:#]', line):
                if not any(bw in line.lower() for bw in blacklist_words):
                    result["name"] = line
                    break
    return result
#Module 2 : Xây dựng Taxonomy & Khớp kỹ năng (Skill Matching Engine).

def extract_skills(self, text: str) -> List[str]:
        """
        Trả về trực tiếp 1 list chứa toàn bộ skills 
        để đối chiếu với model .pkl ở Tầng 2.
        """
        results: Set[str] = set()
        if not text: 
            return []

        doc = self.nlp(text)
        for match_id, start, end in self.matcher(doc):
            canonical_name = self.nlp.vocab.strings[match_id]
            # Nhét thẳng vào 1 set để tự động lọc trùng
            results.add(canonical_name)

        # Trả về 1 list đã sắp xếp theo thứ tự alphabet
        return sorted(list(results))

class SkillExtractor:
    """Module 2: Khớp kỹ năng bằng NLP SpaCy và trả về 1 List duy nhất"""
    def __init__(self):
        self.nlp = spacy.blank("en")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # Từ điển Taxonomy chuẩn hóa kỹ năng
        self.taxonomy = {
            "languages": {
                "Python": ["python", "python3", "py"], "SQL": ["sql", "t-sql", "pl/sql"],
                "C++": ["c++", "cpp"], "Java": ["java"], "R": ["r programming"],
                "JavaScript": ["javascript", "js"], "TypeScript": ["typescript", "ts"], "C#": ["c#", "csharp"]
            },
            "libraries": {
                "Pandas": ["pandas"], "NumPy": ["numpy"], "Scikit-learn": ["scikit-learn", "sklearn"],
                "TensorFlow": ["tensorflow", "tf"], "PyTorch": ["pytorch", "torch"], "Keras": ["keras"],
                "Matplotlib": ["matplotlib"], "Seaborn": ["seaborn"], "React Native": ["react native"]
            },
            "databases": {
                "MySQL": ["mysql"], "PostgreSQL": ["postgresql", "postgres"],
                "SQL Server": ["sql server", "ms sql"], "MongoDB": ["mongodb", "mongo"],
                "Redis": ["redis"], "SQLite": ["sqlite"]
            },
            "tools_and_concepts": {
                "Excel": ["excel", "pivot table"], "Power BI": ["power bi"], "Tableau": ["tableau"],
                "Machine Learning": ["machine learning", "ml", "học máy"],
                "Deep Learning": ["deep learning", "dl"], "OOP": ["oop", "hướng đối tượng"],
                "Git/GitHub": ["git", "github", "gitlab"], "Data Cleaning": ["data cleaning"]
            }
        }
        self.canonical_to_category: Dict[str, str] = {}
        self._build_matcher()

    def _build_matcher(self):
        for category, canonical_dict in self.taxonomy.items():
            for canonical_name, aliases in canonical_dict.items():
                self.canonical_to_category[canonical_name] = category
                patterns = [self.nlp.make_doc(text) for text in aliases]
                self.matcher.add(canonical_name, patterns)

    def extract_skills(self, text: str) -> List[str]:
        """
        Bản cải tiến: Trả về trực tiếp 1 list chứa toàn bộ skills 
        để đối chiếu với model .pkl ở Tầng 2.
        """
        results: Set[str] = set()
        if not text: 
            return []

        doc = self.nlp(text)
        for match_id, start, end in self.matcher(doc):
            canonical_name = self.nlp.vocab.strings[match_id]
            # Nhét thẳng vào set để tự động lọc trùng
            results.add(canonical_name)

        # Trả về 1 list đã sắp xếp theo thứ tự alphabet
        return sorted(list(results))

class ExperienceExtractor:
    """Module 2.3: Khai phá số năm kinh nghiệm với thuật toán Merge Intervals"""
    def __init__(self, current_year: int = None):
        # Tự động lấy năm hiện tại
        self.current_year = current_year or datetime.now().year
        
        self.exp_header_pattern = re.compile(
            r'(?:^|\n)\s*(?:WORK\s+EXPERIENCE|EXPERIENCE|EMPLOYMENT|KINH\s*NGHIỆM(?:\s*LÀM\s*VIỆC)?|PROJECTS?|DỰ\s*ÁN)\b', re.IGNORECASE
        )
        self.next_header_pattern = re.compile(
            r'(?:^|\n)\s*(?:EDUCATION|HỌC\s*VẤN|SKILLS?|KỸ\s*NĂNG|CERTIFICATIONS?|CHỨNG\s*CHỈ|ACTIVITIES|HOẠT\s*ĐỘNG|AWARDS)\b', re.IGNORECASE
        )

    def _extract_exp_section(self, text: str) -> str:
        start_match = self.exp_header_pattern.search(text)
        if not start_match: return ""
        start_idx = start_match.end()
        end_match = self.next_header_pattern.search(text, pos=start_idx)
        end_idx = end_match.start() if end_match else len(text)
        return text[start_idx:end_idx].strip()

    def _classify_level(self, years: float) -> str:
        if years < 1.0: return "Intern / Fresher"
        elif 1.0 <= years < 3.0: return "Junior"
        elif 3.0 <= years < 5.0: return "Mid-Level"
        return "Senior"

    def extract_experience(self, text: str, raw_exp_str: Optional[str] = None) -> Dict[str, Any]:
        """Trả về Dictionary chứa 2 key: 'years' và 'level'"""
        if raw_exp_str:
            num_match = re.search(r'(\d+(?:\.\d+)?)', raw_exp_str)
            if num_match: return {"years": float(num_match.group(1)), "level": self._classify_level(float(num_match.group(1)))}

        search_target = self._extract_exp_section(text) or text

        year_explicit = re.search(r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?|năm)', search_target, re.IGNORECASE)
        if year_explicit: return {"years": float(year_explicit.group(1)), "level": self._classify_level(float(year_explicit.group(1)))}

        # Thuật toán Merge Intervals chống gối đầu thời gian
        date_ranges = re.findall(r'\b(20\d{2})\s*[-–—tođến]+\s*(20\d{2}|present|nay|hiện\s*tại)\b', search_target, re.IGNORECASE)
        if date_ranges:
            intervals = []
            for start_str, end_str in date_ranges:
                start_y = int(start_str)
                end_y = self.current_year if any(k in end_str.lower() for k in ["present", "nay", "hiện"]) else int(end_str)
                if start_y <= end_y: intervals.append([start_y, end_y])
            
            if intervals:
                intervals.sort(key=lambda x: x[0])
                merged = [intervals[0]]
                for current in intervals[1:]:
                    previous = merged[-1]
                    if current[0] <= previous[1]: previous[1] = max(previous[1], current[1])
                    else: merged.append(current)
                
                total_years = sum(end - start for start, end in merged)
                return {"years": float(total_years), "level": self._classify_level(float(total_years))}

        lower_text = text.lower()
        if any(w in lower_text for w in ["intern", "fresher", "sinh viên"]): return {"years": 0.0, "level": "Intern / Fresher"}
        if "senior" in lower_text: return {"years": 5.0, "level": "Senior"}
        if "junior" in lower_text: return {"years": 1.0, "level": "Junior"}

        return {"years": 0.0, "level": "Intern / Fresher"}
class CVParserPipeline:
    
    def __init__(self):
       # các công cụ bóc tách, bỏ phần xuất file JSON
        self.ingestor = CVDataIngestor()
        self.skill_extractor = SkillExtractor()
        self.exp_extractor = ExperienceExtractor()
        
    def process_single_cv(self, cv_id: str, text: str) -> dict:
        # Bóc tách để lấy raw_exp_str (nếu có)
        contact_info = extract_contact_and_labeled_fields(text)
        
        # CỘT 1: Lấy List skills (Module 2.2)
        skills_list = self.skill_extractor.extract_skills(text)
        
        # CỘT 2 & 3: Lấy Level và Years (Module 2.3)
        experience = self.exp_extractor.extract_experience(text, contact_info.get("raw_experience_str"))
        
        # Trả về dict phẳng để chuyển thành DataFrame
        return {
            "candidate_id": cv_id,
            "skills": skills_list,              # 1 List các skill
            "level": experience.get("level"),   # Cấp độ
            "years": experience.get("years")    # Số năm kinh nghiệm
        }

    def run(self, input_path: str) -> pd.DataFrame:
        """
        Chạy pipeline và trả thẳng (return) về Pandas DataFrame cho Tầng 2.
        """
        print(f" Đang nạp dữ liệu từ: {input_path}...")
        raw_cvs = self.ingestor.run(input_path)
        if not raw_cvs:
            print("Không tìm thấy CV nào hợp lệ!")
            return pd.DataFrame() # Trả về DataFrame rỗng nếu lỗi

        print(f"Đã nạp {len(raw_cvs)} CV. Ép xung đa luồng (Multiprocessing)...")
        final_results = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.process_single_cv, cv_id, text): cv_id for cv_id, text in raw_cvs}
            for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                try:
                    final_results.append(future.result())
                except Exception as e:
                    print(f"Lỗi khi xử lý CV: {e}")

                if i % 100 == 0 or i == len(raw_cvs):
                    print(f"  -> Đã bóc tách {i}/{len(raw_cvs)} CV...")

        # GHÉP 3 CỘT VÀO LÀM 1: Tạo DataFrame hoàn chỉnh
        df_final = pd.DataFrame(final_results)
        
        print(f" XONG ! Đã return DataFrame với shape: {df_final.shape}")
        return df_final
