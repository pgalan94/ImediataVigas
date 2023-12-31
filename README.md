# ImediataVigas

## Descrição:

Este aplicativo foi desenvolvido como parte de um trabalho de conclusão de curso de engenharia civil pela Universidade Federal de São Carlos. Trata-se de uma calculadora de flechas imediatas, para vigas biapoiada e concreto armado. O programa não verifica se as solicitações na viga ultrapassam a resistência máxima do elemento estrutural.

Para realização dos cálculos utilizando o método dos elementos finitos, este script utiliza a biblioteca [anaStruct](https://github.com/ritchie46/anaStruct).

## Notas: 
Para Algumas atualizações no código foram realizadas, pois atualizações da
biblioteca pandas tornaram o código incompatível. Como o Artigo foi
publicado sem as versões necessárias para correta utilização, e para
garantir o funcionamento do projeto com a versão atual do pandas,
as utilizações da função `pd.DataFrame().append(df)` foram corretamente
substituidas por `pd.concat([DataFrame, df])`

## Links:

Artigo do TCC publicado: https://repositorio.ufscar.br/handle/ufscar/16716

## Instalação:

Versão do Python recomendada: *3.11.4*.

No windows, utilizando esta versão do Python, as dependências podem ser instaladas rodando o script `install.bat`

1. Clone ou baixe os arquivos deste repositório.

` git clone https://github.com/pgalan94/ImediataVigas.git `

2. Instalar dependências

` pip install -r requirements.txt `

3. Rodar o script

` python standalone.py `
