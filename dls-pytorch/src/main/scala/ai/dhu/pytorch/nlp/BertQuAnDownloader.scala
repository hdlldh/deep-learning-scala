package ai.dhu.pytorch.nlp

import ai.djl.training.util.{DownloadUtils, ProgressBar}

object BertQuAnDownloader {

  def main(args: Array[String]): Unit = {

    DownloadUtils.download(
      "https://djl-ai.s3.amazonaws.com/mlrepo/model/nlp/question_answer/ai/djl/pytorch/bertqa/0.0.1/bert-base-uncased-vocab.txt.gz",
      "build/pytorch/bert-qa/vocab.txt",
      new ProgressBar)

    DownloadUtils.download(
      "https://djl-ai.s3.amazonaws.com/mlrepo/model/nlp/question_answer/ai/djl/pytorch/bertqa/0.0.1/trace_bertqa.pt.gz",
      "build/pytorch/bert-qa/bert-qa.pt",
      new ProgressBar)
  }

}
