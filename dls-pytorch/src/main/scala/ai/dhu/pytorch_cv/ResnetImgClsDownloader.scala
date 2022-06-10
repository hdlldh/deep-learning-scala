package ai.dhu.pytorch_cv

import ai.djl.training.util.{DownloadUtils, ProgressBar}

object ResnetImgClsDownloader {

  def main(args: Array[String]): Unit = {
    DownloadUtils.download(
      "https://djl-ai.s3.amazonaws.com/mlrepo/model/cv/image_classification/ai/djl/pytorch/resnet/0.0.1/traced_resnet18.pt.gz",
      "build/pytorch/resnet18/resnet18.pt",
      new ProgressBar
    )

    DownloadUtils.download(
      "https://djl-ai.s3.amazonaws.com/mlrepo/model/cv/image_classification/ai/djl/pytorch/synset.txt",
      "build/pytorch/resnet18/synset.txt",
      new ProgressBar
    )

  }

}
