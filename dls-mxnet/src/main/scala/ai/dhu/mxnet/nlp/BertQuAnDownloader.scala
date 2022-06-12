package ai.dhu.mxnet.nlp

import ai.djl.training.util.{DownloadUtils, ProgressBar}

object BertQuAnDownloader {

  def main(args: Array[String]): Unit = {
    DownloadUtils.download(
      "https://djl-ai.s3.amazonaws.com/mlrepo/model/nlp/question_answer/ai/djl/mxnet/bertqa/vocab.json",
      "build/mxnet/bert-qa/vocab.json",
      new ProgressBar
    )

    DownloadUtils.download(
      "https://djl-ai.s3.amazonaws.com/mlrepo/model/nlp/question_answer/ai/djl/mxnet/bertqa/0.0.1/static_bert_qa-symbol.json",
      "build/mxnet/bert-qa/bert-qa-symbol.json",
      new ProgressBar
    )

    DownloadUtils.download(
      "https://djl-ai.s3.amazonaws.com/mlrepo/model/nlp/question_answer/ai/djl/mxnet/bertqa/0.0.1/static_bert_qa-0002.params.gz",
      "build/mxnet/bert-qa/bert-qa-0000.params",
      new ProgressBar)
  }
}
