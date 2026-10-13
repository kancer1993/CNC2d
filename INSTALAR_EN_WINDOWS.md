# Cómo ejecutar Image2CNC en Windows (PowerShell)

## 1. Comprueba qué archivos tienes en la carpeta
```powershell
cd C:\Users\JMC\Desktop\workspace
dir
```
Deben aparecer como mínimo:
- `app.py`
- `converter.py`
- `requirements.txt`
- la carpeta `templates` (dentro debe estar `index.html`)

Si NO aparecen, tu copia local está incompleta: vuelve a descargar/copiar TODOS
los archivos de la aplicación (incluida la carpeta `templates`).

## 2. Instala las dependencias
```powershell
pip install -r requirements.txt
```
(El aviso de que hay una versión nueva de pip es inofensivo; puedes ignorarlo.)

## 3. Ejecuta la aplicación
```powershell
python app.py
```
Verás algo como:
```
 * Running on http://127.0.0.1:5000
```

## 4. Abre la interfaz
Entra en tu navegador en: http://localhost:5000

Arrastra una imagen, ajusta parámetros y pulsa **Convertir** para descargar
el DXF y/o el G-code (.nc).

---

## Errores frecuentes

### "can't open file '...app.py': No such file or directory"
Significa que `app.py` no está en la carpeta donde te encuentras. Solución:
```powershell
cd C:\Users\JMC\Desktop\workspace   # ve a la carpeta correcta
dir                                 # verifica que app.py exista ahí
python app.py
```

### Ruta con `/c:/...` no reconocida en PowerShell
Esa ruta es de Git-Bash, no de PowerShell. En PowerShell usa siempre rutas
`C:\...` con backslash. Si quieres usar un Python concreto:
```powershell
& "C:\Users\JMC\Desktop\workspace\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& "C:\Users\JMC\Desktop\workspace\.venv\Scripts\python.exe" app.py
```

### "pip no se reconoce"
Usa `python -m pip install -r requirements.txt` en su lugar.

### El puerto 5000 está ocupado
Cambia el puerto al final de `app.py` (`app.run(port=5000)` → otro número) o
cierra el programa que lo esté usando.
