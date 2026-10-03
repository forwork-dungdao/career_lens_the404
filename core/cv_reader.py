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
                # CẢI TIẾN: Nhóm theo tọa độ X (cột) để chống vỡ layout CV 2 cột
                text_blocks.sort(key=lambda b: (round(b[0] / 200), b[1]))
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
# TẦNG 2: TRÍCH XUẤT THÔNG TIN (EXTRACTION)
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

    # CẢI TIẾN: Bắt số điện thoại quốc tế mở rộng (Mỹ, Anh, VN...)
    phone_match = re.search(r'(?:(?:\+|00)\d{1,3}[\s\-\.]?)?(?:\(?\d{2,4}\)?[\s\-\.]?)?[\d\-\.\s]{7,15}\b', text)
    if phone_match: 
        result["phone"] = re.sub(r'[^\d\+]', '', phone_match.group(0))

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
    def __init__(self, skills_path: str = None):
        self.nlp = spacy.blank("en")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # Từ điển kỹ năng IT mở rộng
        self.taxonomy = {
            # Ngôn ngữ lập trình
            "Python": ["python", "python3", "py"],
            "JavaScript": ["javascript", "js", "es6", "es2015", "vanilla js", "ecmascript"],
            "TypeScript": ["typescript", "ts"],
            "Java": ["java", "core java", "j2ee", "spring boot", "spring", "spring mvc", "hibernate"],
            "C#": ["c#", "csharp", ".net", "asp.net", ".net core", "dotnet"],
            "C++": ["c++", "cpp", "c/c++"],
            "C": ["c language", "ansi c"],
            "PHP": ["php", "laravel", "symfony", "codeigniter"],
            "Go": ["go", "golang"],
            "Ruby": ["ruby", "ruby on rails", "rails"],
            "Rust": ["rust", "rustlang"],
            "Kotlin": ["kotlin"],
            "Swift": ["swift", "swiftui"],
            "Scala": ["scala"],
            "R": ["r language", "r programming", "rstudio"],
            "Matlab": ["matlab"],
            "Shell": ["bash", "shell script", "shell", "powershell", "zsh"],
            
            # Data & AI/ML
            "Machine Learning": ["machine learning", "ml", "học máy"],
            "Deep Learning": ["deep learning", "dl", "neural networks", "neural network"],
            "NLP": ["nlp", "natural language processing", "xử lý ngôn ngữ tự nhiên"],
            "Computer Vision": ["computer vision", "cv", "image recognition", "object detection"],
            "Data Analysis": ["data analysis", "data analytics", "phân tích dữ liệu"],
            "Data Science": ["data science", "khoa học dữ liệu"],
            "Data Engineering": ["data engineering", "data pipeline"],
            "Pandas": ["pandas"],
            "NumPy": ["numpy"],
            "Scikit-learn": ["scikit-learn", "sklearn"],
            "TensorFlow": ["tensorflow", "tf", "keras"],
            "PyTorch": ["pytorch", "torch"],
            "Spark": ["spark", "pyspark", "apache spark"],
            "Hadoop": ["hadoop", "hdfs", "mapreduce"],
            "Power BI": ["power bi", "powerbi"],
            "Tableau": ["tableau"],
            "ETL": ["etl", "extract transform load"],
            "Airflow": ["airflow", "apache airflow"],
            
            # Database
            "SQL": ["sql", "mysql", "postgresql", "t-sql", "pl/sql", "sql server", "mariadb"],
            "NoSQL": ["nosql", "mongodb", "cassandra", "redis", "dynamodb", "couchdb"],
            "Elasticsearch": ["elasticsearch", "elastic search", "elk"],
            "Oracle": ["oracle", "oracle db"],
            "SQLite": ["sqlite"],
            
            # Frontend & Mobile
            "React": ["react", "reactjs", "react.js", "react hooks"],
            "React Native": ["react native"],
            "Angular": ["angular", "angularjs", "angular.js"],
            "Vue": ["vue", "vuejs", "vue.js", "nuxt", "nuxtjs"],
            "Next.js": ["next.js", "nextjs", "next"],
            "HTML/CSS": ["html", "html5", "css", "css3"],
            "Tailwind": ["tailwind", "tailwindcss", "tailwind css"],
            "Bootstrap": ["bootstrap"],
            "SASS/SCSS": ["sass", "scss"],
            "jQuery": ["jquery"],
            "Flutter": ["flutter", "dart"],
            "Android": ["android", "android studio"],
            "iOS": ["ios", "xcode", "objective-c"],
            "Figma": ["figma"],
            
            # Backend & API
            "Node.js": ["node.js", "nodejs", "node", "express", "expressjs"],
            "Django": ["django", "django rest"],
            "FastAPI": ["fastapi", "fast api"],
            "Flask": ["flask"],
            "RESTful API": ["restful api", "rest api", "restful", "api design"],
            "GraphQL": ["graphql"],
            "gRPC": ["grpc"],
            "Microservices": ["microservices", "micro services", "microservice"],
            
            # Cloud & DevOps
            "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda", "sagemaker"],
            "Azure": ["azure", "microsoft azure"],
            "GCP": ["gcp", "google cloud", "google cloud platform", "bigquery"],
            "Docker": ["docker", "containerization", "container"],
            "Kubernetes": ["kubernetes", "k8s"],
            "Git": ["git", "github", "gitlab", "bitbucket", "version control"],
            "CI/CD": ["ci/cd", "cicd", "jenkins", "github actions", "gitlab ci", "circleci"],
            "Terraform": ["terraform", "iac", "infrastructure as code"],
            "Ansible": ["ansible"],
            "Linux": ["linux", "ubuntu", "centos", "redhat"],
            "Nginx": ["nginx"],
            
            # Testing & QA
            "Unit Testing": ["unit test", "unit testing", "jest", "pytest", "mocha"],
            "Selenium": ["selenium", "webdriver"],
            "Cypress": ["cypress"],
            
            # Khác
            "Agile": ["agile", "scrum", "kanban", "sprint"],
            "Jira": ["jira", "atlassian"],
            "RabbitMQ": ["rabbitmq", "message queue"],
            "Kafka": ["kafka", "apache kafka"],
            "WebSocket": ["websocket", "socket.io"],
            "OAuth": ["oauth", "oauth2", "jwt", "json web token"],
            "Blockchain": ["blockchain", "smart contract", "solidity", "web3"],
        }
        
        # Ưu tiên load từ file .pkl nếu có
        if skills_path and Path(skills_path).exists():
            try:
                mlb = joblib.load(skills_path)
                skills = [str(skill) for skill in mlb.classes_]
                patterns = [self.nlp.make_doc(skill) for skill in skills if skill.strip()]
                self.matcher.add("skills", patterns)
                return
            except Exception:
                pass
        
        # Nạp taxonomy vào spaCy PhraseMatcher
        for canonical_name, aliases in self.taxonomy.items():
            patterns = [self.nlp.make_doc(text) for text in aliases]
            self.matcher.add(canonical_name, patterns)

    def extract_skills(self, text: str) -> List[str]:
        if not text: return []
        # Làm sạch dấu câu để chống dính chữ
        clean_text = re.sub(r'[,|/\\;:]', ' ', text)
        results = set()
        doc = self.nlp(clean_text)
        for match_id, start, end in self.matcher(doc):
            string_id = self.nlp.vocab.strings[match_id]
            if string_id in self.nlp.vocab.strings:
                results.add(string_id)
            else:
                results.add(doc[start:end].text)
        return sorted(list(results))

class ExperienceExtractor:
    def __init__(self):
        self.current_year = datetime.now().year
        self.exp_header = re.compile(r'(?:WORK\s+EXPERIENCE|EXPERIENCE|KINH\s*NGHIỆM\s*LÀM\s*VIỆC|KINH\s*NGHIỆM|LỊCH\s*SỬ\s*LÀM\s*VIỆC|EMPLOYMENT)\b', re.IGNORECASE)
        self.next_header = re.compile(r'(?:EDUCATION|HỌC\s*VẤN|SKILLS?|KỸ\s*NĂNG|PROJECTS?|DỰ\s*ÁN|CERTIFICATES?|CHỨNG\s*CHỈ)\b', re.IGNORECASE)

    def _classify_level(self, years: int, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ["senior", "lead", "manager", "principal", "staff"]): return "Senior"
        if any(w in text_lower for w in ["junior", "associate"]): return "Junior"
        if any(w in text_lower for w in ["intern", "fresher", "trainee", "thực tập"]): return "Intern/Fresher"

        if years < 1: return "Intern/Fresher"
        elif 1 <= years < 3: return "Junior"
        elif 3 <= years < 5: return "Mid-Level"
        return "Senior"

    def extract_experience(self, text: str, raw_exp_str: str = None) -> Dict[str, Any]:
        now = datetime.now()
        current_year = now.year
        current_month = now.month

        # Bắt số năm trực tiếp (VD: "3 years of experience")
        search_area = (raw_exp_str + " \n " + text) if raw_exp_str else text
        year_explicit = re.search(r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?|năm)(?:\s*(?:of\s*)?(?:experience|kinh\s*nghiệm))', search_area, re.IGNORECASE)
        if year_explicit:
            yrs = int(round(float(year_explicit.group(1))))
            return {"years": yrs, "level": self._classify_level(yrs, search_area)}

        # Cô lập vùng kinh nghiệm để quét ngày tháng
        exp_section = text
        exp_match = self.exp_header.search(text)
        if exp_match:
            start_idx = exp_match.end()
            next_match = self.next_header.search(text[start_idx:])
            if next_match:
                exp_section = text[start_idx : start_idx + next_match.start()]
            else:
                exp_section = text[start_idx:]

        # Quét khoảng ngày tháng trong vùng kinh nghiệm
        date_ranges = re.findall(
            r'\b(?:(\d{1,2})[/.\-])?(20\d{2})\s*[-–—tođến]+\s*(?:(\d{1,2})[/.\-])?(20\d{2}|present|nay|hiện\s*tại)\b',
            exp_section, re.IGNORECASE
        )

        if date_ranges:
            intervals = []
            for start_m, start_y, end_m, end_y_str in date_ranges:
                s_y = int(start_y)
                s_m = int(start_m) if start_m else 1

                if any(k in end_y_str.lower() for k in ["present", "nay", "hiện"]):
                    e_y = current_year
                    e_m = current_month
                else:
                    e_y = int(end_y_str)
                    e_m = int(end_m) if end_m else 12

                start_abs = s_y * 12 + s_m
                end_abs = e_y * 12 + e_m

                if start_abs <= end_abs:
                    intervals.append([start_abs, end_abs])

            if intervals:
                # Merge Intervals để tránh đếm trùng
                intervals.sort(key=lambda x: x[0])
                merged = [intervals[0]]
                for current in intervals[1:]:
                    previous = merged[-1]
                    if current[0] <= previous[1]:
                        previous[1] = max(previous[1], current[1])
                    else:
                        merged.append(current)

                total_months = sum((end - start + 1) for start, end in merged)
                total_years = int(round(total_months / 12.0))
                return {"years": total_years, "level": self._classify_level(total_years, exp_section)}

        return {"years": 0, "level": self._classify_level(0, exp_section)}

# =====================================================================
# TẦNG 3: ĐIỀU PHỐI (ORCHESTRATION)
# =====================================================================
# Biến toàn cục cho multiprocessing worker
worker_skill_extractor = None
worker_exp_extractor = None

def init_worker():
    global worker_skill_extractor, worker_exp_extractor
    worker_skill_extractor = SkillExtractor()
    worker_exp_extractor = ExperienceExtractor()

def process_single_cv_task(cv_data: Tuple[str, str]) -> dict:
    """Hàm xử lý 1 CV - dùng cho cả single-thread và multiprocessing."""
    cv_id, text = cv_data
    contact_info = extract_contact_and_labeled_fields(text)
    
    # Dùng worker globals nếu có (multiprocessing), nếu không tự tạo (single-thread)
    skill_ext = worker_skill_extractor or SkillExtractor()
    exp_ext = worker_exp_extractor or ExperienceExtractor()
    
    skills_list = skill_ext.extract_skills(text)
    experience = exp_ext.extract_experience(text, contact_info.get("raw_experience_str"))
    
    # ĐỒNG BỘ KEY VỚI app.py: "skill" (list), "năm kinh nghiệm", "level công việc"
    return {
        "skill": skills_list,
        "năm kinh nghiệm": experience.get("years"),
        "level công việc": experience.get("level")
    }

class CVParserPipeline:
    def __init__(self):
        self.ingestor = CVDataIngestor()

    def run(self, input_path: str) -> pd.DataFrame:
        print(f"Đang nạp dữ liệu từ: {input_path}...")
        raw_cvs = self.ingestor.run(input_path)
        
        if not raw_cvs:
            print("Không tìm thấy CV nào hợp lệ!")
            return pd.DataFrame()

        final_results = []

        if len(raw_cvs) == 1:
            # Single CV (Streamlit upload) → chạy trực tiếp, không cần ProcessPool
            try:
                final_results.append(process_single_cv_task(raw_cvs[0]))
            except Exception as e:
                print(f"Lỗi khi xử lý CV: {e}")
        else:
            # Nhiều CV → dùng ProcessPoolExecutor
            print(f"Đã nạp {len(raw_cvs)} CV. Đang xử lý đa tiến trình...")
            with concurrent.futures.ProcessPoolExecutor(initializer=init_worker) as executor:
                futures = [executor.submit(process_single_cv_task, data) for data in raw_cvs]
                for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    try:
                        final_results.append(future.result())
                    except Exception as e:
                        print(f"Lỗi khi xử lý CV: {e}")
                    if i % 10 == 0 or i == len(raw_cvs):
                        print(f"  -> Đã bóc tách {i}/{len(raw_cvs)} CV...")

        df_final = pd.DataFrame(final_results)
        print(f"XONG! Đã return DataFrame với shape: {df_final.shape}")
        return df_final


# =====================================================================
# KHỐI CHẠY KIỂM THỬ
# =====================================================================
if __name__ == "__main__":
    INPUT_FOLDER = r"/content/drive/MyDrive/HACKATHON raw/Dũng/20_cv_doi_chieu_individual_pdfs"
    
    if os.path.exists(INPUT_FOLDER):
        pipeline = CVParserPipeline()
        df_cv = pipeline.run(INPUT_FOLDER)
        
        if not df_cv.empty:
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.width', 1000)
            
            print("\n✅ Đã tạo DataFrame thành công trên RAM!")
            print(df_cv)
    else:
        print(f"Đường dẫn không tồn tại: {INPUT_FOLDER}")