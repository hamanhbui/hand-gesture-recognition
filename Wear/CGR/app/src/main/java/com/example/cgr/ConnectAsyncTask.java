package com.example.cgr;

import android.os.AsyncTask;
import android.view.View;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.ServerSocket;
import java.net.Socket;
public class ConnectAsyncTask extends AsyncTask<Void,Boolean,Void>{
    private MainActivity contextParent;
    public ConnectAsyncTask(MainActivity contextParent){
        this.contextParent=contextParent;
    }
    @Override
    protected void onPreExecute(){
        super.onPreExecute();
    }
    @Override
    protected Void doInBackground(Void... params) {
        try {
            ServerSocket serverSocket=new ServerSocket(5556);
            while(true) {
                Socket socket=serverSocket.accept();
                ObjectInputStream ois=new ObjectInputStream(socket.getInputStream());
                String ping= (String) ois.readObject();
                if(ping.equals("Ping!")){
                    ObjectOutputStream oos=new ObjectOutputStream(socket.getOutputStream());
                    oos.writeObject("Accepted!");
                    oos.flush();
                    contextParent.ipMobile=socket.getInetAddress();
                    serverSocket.close();
                    publishProgress(true);
                    contextParent.createSocket();
                    break;
                }
            }
        }catch(Exception e){
            System.out.println(e.toString());
        }
        return null;
    }

    @Override
    protected void onProgressUpdate(Boolean... values) {
        super.onProgressUpdate(values);
        boolean signal=values[0];
        if(signal) {
            contextParent.mTextView.setText("Start recording....");
        }
    }

    @Override
    protected void onPostExecute(Void aVoid) {
        super.onPostExecute(aVoid);
    }
}
