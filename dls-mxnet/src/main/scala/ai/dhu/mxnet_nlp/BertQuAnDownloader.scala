package ai.dhu.mxnet_nlp

import ai.djl.training.util.{DownloadUtils, ProgressBar}

object BertQuAnDownloader {

  def main(args: Array[String]): Unit = {
    DownloadUtils.download(
      "https://djl-ai.s3.amazonaws.com/mlrepo/model/nlp/question_answer/ai/djl/mxnet/bertqa/vocab.json",
      "build/mxnet/bertqa/vocab.json",
      new ProgressBar
    )

    DownloadUtils.download(
      "https://djl-ai.s3.amazonaws.com/mlrepo/model/nlp/question_answer/ai/djl/mxnet/bertqa/0.0.1/static_bert_qa-symbol.json",
      "build/mxnet/bertqa/bertqa-symbol.json",
      new ProgressBar
    )

    DownloadUtils.download(
      "https://djl-ai.s3.amazonaws.com/mlrepo/model/nlp/question_answer/ai/djl/mxnet/bertqa/0.0.1/static_bert_qa-0002.params.gz",
      "build/mxnet/bertqa/bertqa-0000.params",
      new ProgressBar)
  }
}
