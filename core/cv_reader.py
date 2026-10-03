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



