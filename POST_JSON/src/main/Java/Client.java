package main.Java;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.util.EntityUtils;
import org.json.JSONObject;

@SuppressWarnings("unused")
public class Client {
    public static void main(String[] args) throws Exception {
        String word = "Japanese";
        int syllable[] = {36, 0, 12, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        JSONObject obj = new JSONObject();
        obj.put("name", word);
        obj.put("syllable", syllable);
        String st = obj.toString();

        StringEntity entity = new StringEntity(st);
        HttpClient httpClient = HttpClientBuilder.create().build();
        HttpPost request = new HttpPost("http://localhost:5000/api/test");
        request.setHeader("Content-type", "application/json");
        request.setEntity(entity);

        HttpResponse response = httpClient.execute(request);
        HttpEntity entity1 = response.getEntity();
        String response1 = EntityUtils.toString(entity1, "utf-8");
        JSONObject result = new JSONObject(response1);
        double percentage = result.getDouble("true");

//        System.out.println(percentage);
    }
}