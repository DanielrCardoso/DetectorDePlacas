# Detector de Placas

Este programa foi desenvolvido para a matéria de Introdução ao processamento de imagens da Universidade de Brasília. O intuito do software é aplicar algumas técnicas aprendidas no decorrer da disciplina para realizar a extração de caracteres de placas de veiculos.

# Preparação do ambiente:
É necessário ter o compilador do python para executar o projeto. É possível realizar a instalação do compilador python a partir da página oficial da linguagem(https://www.python.org/downloads/). Durante o processo de desenvolvimento foi adotado a versão 3.10.5.

Instale as bibliotecas necessarias:

```  
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
Obs: Outras opções de instalção do pytorch podem ser encontradas no site oficial: https://pytorch.org/get-started/locally/

``` 
pip install PyQt5
``` 
``` 
pip install cv2
```  
```  
pip install numpy
```  
```  
pip install imutils
``` 
```   
pip install easyocr
```  

# Execução do programa
No terminal, clone o projeto:  
```  
git clone https://github.com/DanielrCardoso/DetectorDePlacas
``` 

Execute o código: 
```  
python TelaIPI.py 
``` 