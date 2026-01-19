import os
from fastapi.templating import Jinja2Templates  
templates = Jinja2Templates(directory="templates")
print("Template directory exists:", os.path.exists("lcd_detec2//templates"))


