# Proactive Deepfake Defense

This is a repository that explain my research.
# 1. Deepfakeとは
    Deepfakes (portmanteau of "deep learning" and "fake") are synthetic media that have been digitally manipulated to replace one person's likeness convincingly with that of another. Deepfakes are the manipulation of facial appearance through deep generative methods.
<br>
    ディープフェイク（deepfake）は、「深層学習（deep learning）」と「偽物（fake）」を組み合わせた混成語（かばん語）で、人工知能にもとづく人物画像合成の技術を指す。「敵対的生成ネットワーク（GANs）」と呼ばれる機械学習技術を使用して、既存の画像と映像を、元となる画像または映像に重ね合わせて（スーパーインポーズ）、結合することで生成される[2]。 既存と元の映像を結合することにより、実際には起こっていない出来事で行動している1人あるいは複数人の偽の映像が生み出されることとなる。
<br>
<br>
*Image 1: Deepfakeの悪用*
<br>
[![Image text](https://github.com/Joe-997/Proactive-Deepfake-Defense/blob/main/img/3.png)](https://www3.nhk.or.jp/news/html/20231104/k10014247171000.html)
<br>

# 2. 研究内容
課題：近年、GANs、VAEs、Diffusionモデルなどの画像生成技術は前例のない成長を遂げ、超リアルな画像出力をもたらしています。しかし、これには誤情報の拡散やデジタル詐欺といった課題が伴っています。
<br>
手法：課題に対処するため、私たちは画像に目に見えないノイズを加える手法を研究しています。このノイズがある画像は、画像生成モデルで処理されると出力が乱れ、誤用から保護する役割を果たします。ノイズを追加するための従来のアルゴリズムは、ホワイトボックス攻撃が必要であり、目的モデルのすべてのパラメータが必要となり、その実用性に制約があります。これに対して、敵対的トレーニングを通じて、セミブラックボックス攻撃の目的を一定程度達成できるモデルを開発することを目指しています。
<br>
*Image 2: Deepfakeーー顔の変更*
<br>
![Image text](https://github.com/Joe-997/Proactive-Deepfake-Defense/blob/main/img/1.png)
<br>
*Image 3: 理想的な結果ーー顔の変更は失敗した*
<br>
![Image text](https://github.com/Joe-997/Proactive-Deepfake-Defense/blob/main/img/2.png)
<br>
