package ai.dhu.pytorch.nlp

import java.nio.file.Paths

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.djl.modality.nlp.DefaultVocabulary
import ai.djl.ndarray.NDList
import ai.djl.translate.{Batchifier, Translator, TranslatorContext}

class HFBertFillMaskTranslator extends Translator[String, Seq[PredictedToken]] {

  private var tokens: Array[String] = _
  private var vocabulary: DefaultVocabulary = _
  private final val tokenizer = HuggingFaceTokenizer.newInstance("bert-base-uncased")
  private final val MaskToken = "[MASK]"
  private final val TopK = 5

  override def prepare(ctx: TranslatorContext): Unit = {
    val path = Paths.get("build/huggingface/fill-mask/pytorch/bert-base-uncased/vocab.txt")
    vocabulary = DefaultVocabulary.builder
      .optMinFrequency(1)
      .addFromTextFile(path)
      .optUnknownToken("[UNK]")
      .build
  }

  override def processInput(ctx: TranslatorContext, input: String): NDList = {
    val encoded = tokenizer.encode(input.toLowerCase().replace(MaskToken.toLowerCase(), MaskToken))
    tokens = encoded.getTokens

    val manager = ctx.getNDManager
    val tokenIds = encoded.getIds
    val attentionMask = encoded.getAttentionMask
    val tokenIdsArray = manager.create(tokenIds)
    val attentionMaskArray = manager.create(attentionMask)
    new NDList(tokenIdsArray, attentionMaskArray)
  }

  override def processOutput(ctx: TranslatorContext, list: NDList): Seq[PredictedToken] = {
    val maskIndex = tokens.zipWithIndex.find(_._1 == MaskToken).map(_._2).getOrElse(-1)
    if (maskIndex == -1) {
      Seq.empty[PredictedToken]
    } else {
      val ndArray = list.get(0)
      val len = ndArray.size(1)

      (1 to TopK).map { i =>
        val out = ndArray.get(maskIndex).argSort().getLong(len - i)
        PredictedToken(vocabulary.getToken(out), ndArray.getFloat(maskIndex, out))
      }
    }
  }

  override def getBatchifier: Batchifier = Batchifier.STACK
}
