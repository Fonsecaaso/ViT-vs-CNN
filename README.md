# Identify plant by photo - CNN vs Transformers

Trabalho de Conclusão de Curso em desenvolvimento. Estou desenvolvendo modelo de rede neural convolucional que classifica a imagem de acordo com a espécie. Estou na fase de montar base de dados, a priori quero usar imagem de flores e árvores brasileiras. Assim que tiver uma base montada irei testar com diferentes modelos de arquiteturas (CNN E Transformers), avaliar performance e comparar resultados. Vejo que durante esse processo meu computador pessoal e o Google Colab free não serão suficientes, então provavelemente eu irei utilizar serviço pago da AWS (Amazon SageMaker)

![](https://media.giphy.com/media/CWx3Qijmkf4746TtGu/giphy.gif)

## Fases

1. Revisão Bibliográfica
2. Criar dataset de plantas, focando em plantas brasileiras e cada classe deve, em primeiro momento, focar apenas em flor ou árvore
3. Testar diferentes modelos de redes convolucionais e transformers e avaliar resultados obtidos

## Revisão Bibliográfica

#### Artigos relevantes no campo de visão computacional:

1. "ImageNet Classification with Deep Convolutional Neural Networks" de Alex Krizhevsky, Ilya Sutskever e Geoffrey Hinton (2012): Este artigo introduziu o modelo de rede neural convolucional AlexNet, que estabeleceu novos recordes de precisão na classificação de imagens em desafios competitivos, como o ImageNet Large Scale Visual Recognition Challenge (ILSVRC).

2. "Very Deep Convolutional Networks for Large-Scale Image Recognition" de Karen Simonyan e Andrew Zisserman (2014): Este artigo apresenta o modelo de rede neural convolucional VGG, que foi um dos primeiros a alcançar alta precisão na classificação de imagens em grandes conjuntos de dados, como o ImageNet.

3. "Going Deeper with Convolutions" de Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke e Andrew Rabinovich (2015): Este artigo introduziu o modelo de rede neural convolucional Inception, que utiliza módulos de convolução com diferentes tamanhos de filtro para melhorar a eficiência computacional e a precisão da classificação de imagens.

4. "Deep Residual Learning for Image Recognition" de Kaiming He, Xiangyu Zhang, Shaoqing Ren e Jian Sun (2016): Este artigo introduziu o modelo de rede neural convolucional ResNet, que utiliza blocos residuais para permitir o treinamento de redes mais profundas e alcançar maior precisão na classificação de imagens.

5. "Mask R-CNN" de Kaiming He, Georgia Gkioxari, Piotr Dollár e Ross Girshick (2017): Este artigo introduziu o modelo de rede neural convolucional Mask R-CNN, que combina detecção de objetos e segmentação semântica em uma única rede neural e alcançou alta precisão em desafios de detecção e segmentação de objetos em imagens.

#### Artigos relevantes na temática de classificação de plantas usando deep learning:

6. "Plant species classification using deep convolutional neural network"

7. "VGG16 for Plant Image Classification with Transfer Learning and Data Augmentation"

8. "Flower classification using deep convolutional neural networks" 

9. "Simplifying VGG-16 for Plant Species Identification"

10. "A comprehensive comparison on current deep learning approaches for plant image classification"

#### Artigos relevantes na temática de Transformers

11. "Attention Is All You Need" (Vaswani et al., 2017) - Este é o paper original que introduziu o Transformer na comunidade de aprendizado profundo. O paper descreve uma arquitetura baseada em autoatendimento (self-attention) que elimina a necessidade de uma convolução em redes neurais seq2seq, permitindo que o modelo se concentre nas partes relevantes da entrada.

![image](https://deepfrench.gitlab.io/deep-learning-project/resources/transformer.png)

12. "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020) - Este é o paper que apresentou o Vision Transformer (ViT), que aplica a arquitetura Transformer diretamente a imagens. O ViT mostrou-se competitivo com as arquiteturas de referência em visão computacional, como a ResNet, e permitiu que a arquitetura Transformer fosse aplicada diretamente a imagens.

![](https://github.com/lucidrains/vit-pytorch/blob/main/images/vit.gif)


13. "Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet" (Han et al., 2021) - Este paper propõe um método mais eficiente de treinamento do ViT, que usa apenas tokens como entrada em vez de patches de imagem, tornando o processo de treinamento mais escalável.

14. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (Liu et al., 2021) - Este paper apresenta o Swin Transformer, uma arquitetura que emprega janelas deslocadas para permitir uma hierarquia de recursos de tamanho crescente, tornando o modelo mais escalável e preciso.

15. "MLP-Mixer: An all-MLP Architecture for Vision" (Tolstikhin et al., 2021) - Este paper propõe uma arquitetura baseada em MLP (Multilayer Perceptron) para imagens que se mostrou tão eficaz quanto o ViT em vários conjuntos de dados de visão, mas sem o alto custo computacional associado ao autoatendimento (self-attention).

#### Links para consulta:

kanban: https://trello.com/b/cyxsrmqp/tcc-vit-vs-cnn

https://www.robots.ox.ac.uk/~vgg/research/flowers/

https://www.kaggle.com/code/milan400/flowers-classification-using-vgg19
