# Sensores Laboratório de Projetos 4
Projeto desenvolvido para disciplina laboratório de projetos 4 no curso de engenharia de sistemas da UFMG.

- lenet_ensemble_main -- Código principal da métodologia, utilizado para dar entrada dos 3 sinais e treinar a rede proposta. É feito um looping responsável pela validação. É um treinamento para cada pessoa presente no dataset (live one subject out).
São utilizados 3 representações de Acelerometro. A primeira é o link acima (temporal), a segunda deve ser gerada utilizado o my_recurrentplot (representação de imagem), a terceira é gerada no código principal, a partir da primeira (spectrogram).

- my_recurrentplot -- Código para gerar a representação da imagem utilizada como entrada no método. Primeiro é necessário rodar para cada dataset, para gerar um arquivo utilizado no código principal.

- tinyResnet -- Rede utilizada por uma das referencias (precisaremos rodar como comparação).

- spec_shufflenet -- Rede utilizaremos como comparação.


link dos datasets: https://drive.google.com/drive/folders/11RnkBeYauWMzTn1zcp9w1XgH9WPa8LkE?usp=sharing
