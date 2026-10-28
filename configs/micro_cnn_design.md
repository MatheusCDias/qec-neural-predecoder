# Micro-CNN local: proposta v1

## Decisão de representação

Não reshapear a lista linear de detectores assumindo uma ordem espacial. Usar
`circuit.get_detector_coordinates()` como fonte da geometria: cada coordenada
é `(x, y, t)`. Para os circuitos atuais, `x` e `y` são múltiplos de 2, e `t`
vai de 0 a `rounds`. O índice linear do detector só identifica qual coluna da
matriz de síndrome deve ser lida; não define vizinhança.

Normalizar para `ix = x / 2`, `iy = y / 2`, `it = t`, e espalhar a síndrome em
um tensor PyTorch channels-first `(batch, channels, time, y, x)` de dimensões
`(B, 5, rounds+1, d+1, d+1)`. Células sem detector recebem zero no canal de
síndrome e zero na máscara de ocupação. Os cinco canais são:

1. evento de síndrome;
2. máscara de detector presente;
3. indicador de borda espacial;
4. indicador de camada temporal inicial/final;
5. sublattice/checkerboard, codificado como `-1/+1` nas células ocupadas.

As dimensões ficam `d=3: (B,5,4,4,4)` e `d=5: (B,5,6,6,6)`. Nos circuitos
Stim inspecionados, isso preserva as posições reais: `d=3` tem 4, 8, 8, 4
detectores por camada e `d=5` tem 12, 24, 24, 24, 24, 12. A máscara distingue
células ausentes daquelas que contêm um detector sem defeito. A construção
deve falhar explicitamente se faltarem coordenadas, houver colisões após a
normalização ou surgirem coordenadas incompatíveis; não deve adivinhar a
geometria.

## Rede e contagem de parâmetros

Usar uma Conv3D rasa com padding 1 e ReLU/GroupNorm:

- `Conv3d(5, 12, kernel_size=3)`;
- `Conv3d(12, 16, kernel_size=3)`;
- `Conv3d(16, 12, kernel_size=3)`.

Isso produz um embedding local de 12 canais por detector. Um cabeçalho MLP
compartilhado pontua cada mecanismo de erro local do DEM: concatena embedding
médio/máximo dos detectores atingidos, extensão `(dx,dy,dt)`, tamanho do
suporte e `-log(p)` do mecanismo; usa `Linear(29,32) -> ReLU -> Linear(32,1)`.
Com biases e GroupNorm, o total é aproximadamente 13.101 parâmetros, abaixo
do limite de 25 mil. O número de mecanismos candidatos varia por circuito,
mas o número de parâmetros não varia.

O conjunto de candidatos é derivado do mesmo DEM usado para compilar o
PyMatching. Na v1, aceitar apenas mecanismos cujo suporte detector tenha
diâmetro local máximo de uma célula em `(ix,iy,it)`; mecanismos fora desse
raio ficam para o MWPM. Mecanismos com um detector são candidatos de borda.
Cada mecanismo conserva sua máscara de detectores e sua máscara de observáveis
do DEM, incluindo suportes correlacionados; isso evita reduzir uma falha
correlacionada a decisões de aresta independentes.

## Target e perda

Preferir prever ativações de mecanismos de erro locais do DEM em vez de prever
apenas “aniquilar” um par de detectores. A aniquilação sem uma correção
associada não define como atualizar os observáveis e pode alterar artificialmente
`p_L`.

Gerar supervisão com `dem.compile_sampler().sample(return_errors=True)`, que
fornece síndrome, observáveis e ativações dos mecanismos amostrados. O target
principal é multi-label: um bit por mecanismo local. A perda proposta é:

`L = L_focal(z_hat, z_local) + 0.25 L_parity + 0.1 L_obs`.

- `L_focal`: focal BCE por mecanismo, com pesos calculados no conjunto de treino
  para compensar os positivos raros;
- `L_parity`: erro de paridade diferenciável das máscaras de detector previstas
  contra as ativações locais alvo. Para probabilidades independentes `q_e`, a
  probabilidade de paridade ímpar de um detector é
  `(1 - product_e(1 - 2*q_e))/2`;
- `L_obs`: perda BCE sobre a paridade prevista das máscaras de observáveis dos
  mecanismos locais alvo.

O decoder usa decisões binárias calibradas em validação (limiar selecionado
para desempenho `p_L`/latência, não em treino). Para as ativações selecionadas
`z`, calcular exatamente em GF(2):

`s_residual = s XOR (z @ H_DEM)` e `o_pre = z @ L_DEM`.

Enviar `s_residual` ao PyMatching e compor `o_final = o_pre XOR o_MWPM`. Essa
construção mantém o contrato do `HybridDecoder`. A validação de sucesso é a
curva `p_L` e a latência end-to-end em circuitos Stim reais; a acurácia de bits
de mecanismo não é métrica final, pois mecanismos equivalentes podem produzir
a mesma síndrome/observável.

## Plano de implementação e riscos

1. Implementar o tensorizador de coordenadas com testes de round-trip para
   `d=3` e `d=5`, incluindo dimensões, ocupação e ordem do detector.
2. Extrair do DEM as máscaras de detector/observável e probabilidades por
   mecanismo; verificar por amostragem que `errors @ H_DEM` e `errors @ L_DEM`
   reproduzem os detectores e observáveis devolvidos pelo sampler.
3. Implementar dataset DEM supervisionado, Micro-CNN e seleção de mecanismos.
4. Comparar contra baseline e identidade usando mesmos circuitos, sementes,
   shots, splits por seed e medição de pré-processamento + rede + MWPM.

O DEM pode ser uma aproximação do ruído do circuito e a classificação de
mecanismos é ambígua sob degenerescência. Portanto, treinar com o DEM e decidir
com a curva em amostras independentes do circuito; não apresentar ganho em
acurácia de mecanismo como ganho de decodificação.

## Leitura cautelosa do baseline atual

A malha exploratória de 10.000 shots aponta uma região de cruzamento finito
aproximadamente entre `p=0.005` e `p=0.01`; ela não determina o threshold com
precisão. Em `p=0.005`, `p_L` medido é 0.0164, 0.0152 e 0.0099 para `d=3,5,7`;
em `p=0.01`, é 0.0583, 0.0795 e 0.0994. Com apenas três distâncias, uma grade
esparsa e poucos shots nos pontos de baixo erro, não declarar ainda
`p_th = 0.005–0.007`. Aumentar shots perto da região de cruzamento e ajustar
scaling de tamanho finito antes de fixar esse número.
