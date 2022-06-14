package ai.dhu.pytorch.nlp

import java.nio.file.Paths

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.ndarray.NDList
import ai.djl.translate.{Batchifier, Translator, TranslatorContext}

class HFBertFillMaskTranslator extends Translator[String, Seq[PredictedToken]] {

  private var vocabulary: DefaultVocabulary = _
  private var tokenizer: HuggingFaceTokenizer = _
  private var tokenList: Array[String] = _
  private final val MaskToken = "[MASK]"
  private final val TopK = 5

  override def prepare(ctx: TranslatorContext): Unit = {
    val path = Paths.get("build/huggingface/fill_mask/pytorch/bert-base-uncased/vocab.txt")
    vocabulary = DefaultVocabulary.builder
      .optMinFrequency(1)
      .addFromTextFile(path)
      .optUnknownToken("[UNK]")
      .build
    tokenizer = HuggingFaceTokenizer.newInstance("bert-base-uncased")
  }

  override def processInput(ctx: TranslatorContext, input: String): NDList = {
    val token = tokenizer.encode(input.toLowerCase().replace(MaskToken.toLowerCase(), MaskToken))
    // get the encoded tokens that would be used in precessOutput
    tokenList = token.getTokens
    // map the tokens(String) to indices(long)

    val manager = ctx.getNDManager
    val indices = tokenList.map(vocabulary.getIndex)
    val attentionMask = token.getAttentionMask.map(i => i)
    val indicesArray = manager.create(indices)
    val attentionMaskArray = manager.create(attentionMask)

    new NDList(indicesArray, attentionMaskArray)
  }

  override def processOutput(ctx: TranslatorContext, list: NDList): Seq[PredictedToken] = {
    val maskIndex = tokenList.zipWithIndex.find(_._1 == MaskToken).map(_._2).getOrElse(-1)
    if (maskIndex == -1) {
      Seq.empty[PredictedToken]
    } else {
      val ndArray = list.get(0)
      val shape = ndArray.getShape
      val len = shape.get(1)

      (1 to TopK).map { i =>
        val out = ndArray.get(maskIndex).argSort().getLong(len - i)
        PredictedToken(vocabulary.getToken(out), ndArray.getFloat(maskIndex, out))
      }
    }
  }

  override def getBatchifier: Batchifier = Batchifier.STACK
}
