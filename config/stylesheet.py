import re
from bin.log import get_logger

log = get_logger("ui")

def get_stylesheet(file_path, theme_index=0):
    with open(file_path, 'r', encoding='utf-8') as f:
        full_content = f.read()

    comment_blocks = re.findall(r'/\*(.*?)\*/', full_content, re.DOTALL)
    themes = []
    for block in comment_blocks:
        if "THEME:" in block:
            lines = block.strip().split('\n')
            current_theme = None
            for line in lines:
                line = line.strip()
                if line.startswith("THEME:"):
                    if current_theme: themes.append(current_theme)
                    current_theme = {"name": line.replace("THEME:", "").strip()}
                elif line.startswith("VAR_"):
                    parts = line.split(':')
                    if len(parts) == 2 and current_theme != None:
                        current_theme[parts[0].strip()] = parts[1].strip()
            if current_theme: themes.append(current_theme)

    css_body = re.sub(r'/\*.*?\*/', '', full_content, flags=re.DOTALL)

    if not themes:
        return css_body

    selected_theme = themes[theme_index] if theme_index < len(themes) else themes[0]
    log.info(f"Selected theme: {selected_theme.get("name")}")

    def var_replacer(match):
        name = match.group(0)
        if name not in selected_theme:
            log.warning(f"Unknown stylesheet variable '{name}'")
        return selected_theme.get(name, "#ff00ff")

    final_css = re.sub(r'VAR_[A-Z0-9_]+', var_replacer, css_body)

    return final_css

def update_state(widget, property_name, value):
    widget.setProperty(property_name, value)
    
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()