# BIICC-2023

## Instalar zstd (necesario para descomprimir archivos):

- sudo apt install zstd

(Si se está usando conda, tener en cuenta dónde se instala zstd.)

## Instalar fasttext:

El código fuente se encuentra aquí:
- https://github.com/facebookresearch/fastText/releases

Hay un tutorial para instalarlo, pero está desactualizado.
- https://fasttext.cc/docs/en/supervised-tutorial.html

Agregar directorio a PATH para no tener que hacer ./.../fasttext:

- Open the .bashrc file using a text editor.
- Go to the end of the file.
- Paste the export syntax at the end of the file. 
        export PATH="/.../fasttext:$PATH"
- Save and exit.
