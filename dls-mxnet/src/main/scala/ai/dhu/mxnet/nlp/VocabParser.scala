package ai.dhu.mxnet.nlp

import ai.djl.util.JsonUtils
import com.google.gson.annotations.SerializedName

import java.io.{IOException, InputStreamReader}
import java.net.URL
import java.nio.charset.StandardCharsets
import java.util

class VocabParser {
  @SerializedName("idx_to_token")
  var idx2token: util.List[String] = _

}

object VocabParser {
  def parseToken(file: URL) = {
    try {
      val is = file.openStream()
      val reader = new InputStreamReader(is, StandardCharsets.UTF_8)
      JsonUtils.GSON.fromJson(reader, classOf[VocabParser]).idx2token
    } catch {
      case e: IOException => throw new IllegalArgumentException("Invalid url: " + file, e)
    }
  }
}