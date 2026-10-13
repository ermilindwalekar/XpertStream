"""Packages vscode-xpert/ into a .vsix (no Node.js needed) and installs it into VS Code."""

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

EXT_DIR = Path(__file__).resolve().parent.parent / "vscode-xpert"
FILES = ["package.json", "extension.js"]

CONTENT_TYPES = """<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension=".json" ContentType="application/json"/>
  <Default Extension=".js" ContentType="application/javascript"/>
  <Default Extension=".vsixmanifest" ContentType="text/xml"/>
</Types>
"""


def manifest(pkg: dict) -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011">
  <Metadata>
    <Identity Language="en-US" Id="{escape(pkg['name'])}" Version="{escape(pkg['version'])}" Publisher="{escape(pkg['publisher'])}"/>
    <DisplayName>{escape(pkg['displayName'])}</DisplayName>
    <Description xml:space="preserve">{escape(pkg['description'])}</Description>
    <Properties>
      <Property Id="Microsoft.VisualStudio.Code.Engine" Value="{escape(pkg['engines']['vscode'])}"/>
    </Properties>
  </Metadata>
  <Installation>
    <InstallationTarget Id="Microsoft.VisualStudio.Code"/>
  </Installation>
  <Dependencies/>
  <Assets>
    <Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json" Addressable="true"/>
  </Assets>
</PackageManifest>
"""


def is_up_to_date(pkg: dict) -> bool:
    installed = Path.home() / ".vscode" / "extensions" / f"{pkg['publisher']}.{pkg['name']}-{pkg['version']}"
    try:
        installed_pkg = json.loads((installed / "package.json").read_text(encoding="utf-8"))
        # VS Code adds __metadata to the installed package.json.
        installed_pkg.pop("__metadata", None)
        return (installed_pkg == pkg
                and (installed / "extension.js").read_bytes() == (EXT_DIR / "extension.js").read_bytes())
    except (OSError, ValueError):
        return False


def main():
    pkg = json.loads((EXT_DIR / "package.json").read_text(encoding="utf-8"))
    if "--force" not in sys.argv and is_up_to_date(pkg):
        print(f"[*] VS Code extension {pkg['name']} {pkg['version']} is up to date.")
        return 0

    vsix = EXT_DIR / f"{pkg['name']}-{pkg['version']}.vsix"
    with zipfile.ZipFile(vsix, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("extension.vsixmanifest", manifest(pkg))
        for name in FILES:
            z.write(EXT_DIR / name, f"extension/{name}")
    print(f"[+] Built {vsix}")

    code = shutil.which("code") or shutil.which("code.cmd")
    if not code:
        print("[!] 'code' command not found. Install manually: Extensions view > ... > Install from VSIX.")
        return 1
    result = subprocess.run([code, "--install-extension", str(vsix), "--force"])
    if result.returncode == 0:
        print("[+] Installed. Reload any open VS Code windows (Ctrl+Shift+P > 'Developer: Reload Window').")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
