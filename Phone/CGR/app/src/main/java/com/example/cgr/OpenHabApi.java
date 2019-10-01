package com.example.cgr;

import com.google.gson.JsonObject;

import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Path;

public interface OpenHabApi {

    @POST("items/{itemname}")
    Call<Void> sendsACommandToAnItem(@Path("itemname") String itemName, @Body RequestBody body);

    @GET("items/{itemname}")
    Call<JsonObject> getASingleItem(@Path("itemname") String itemname);
}
