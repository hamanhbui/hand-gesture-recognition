package com.example.cgr;

import android.content.Context;
import android.os.AsyncTask;
import android.os.PowerManager;
import android.support.v4.app.FragmentTransaction;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.Toast;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;

public class MainActivity extends AppCompatActivity {
    private PowerManager.WakeLock mWakeLock;
    private ActivityInference mActivityInference;
    private ConnectAsyncTask mConnectAsyncTask;
    private String mWatchIP;
    private Button mBtnConnectToRecord;
    private Button mBtnConnectToPredict;
    private Button mBtnConnectToRecord2;
    private EditText mEdtIp;
    private ProgressBar mProgressBar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        mWakeLock = pm.newWakeLock(PowerManager.SCREEN_DIM_WAKE_LOCK, "My Tag");
        mWakeLock.acquire();

        mActivityInference = new ActivityInference(getApplicationContext());
        initViews();
    }
    @Override
    public void onDestroy() {
        this.mWakeLock.release();
        super.onDestroy();
    }

    private void initViews() {
        mProgressBar = findViewById(R.id.progressBar);
        mBtnConnectToPredict = findViewById(R.id.btn_predict);
        mBtnConnectToRecord = findViewById(R.id.btn_record);
        mBtnConnectToRecord2=findViewById(R.id.button18);
        mEdtIp = findViewById(R.id.editText);

        mBtnConnectToPredict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mWatchIP = mEdtIp.getText().toString();
                mConnectAsyncTask = new ConnectAsyncTask(mWatchIP, 0);
                mConnectAsyncTask.execute();
            }
        });

        mBtnConnectToRecord.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mWatchIP = mEdtIp.getText().toString();
                mConnectAsyncTask = new ConnectAsyncTask(mWatchIP, 1);
                mConnectAsyncTask.execute();
            }
        });

        mBtnConnectToRecord2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mWatchIP = mEdtIp.getText().toString();
                mConnectAsyncTask = new ConnectAsyncTask(mWatchIP, 2);
                mConnectAsyncTask.execute();
            }
        });
    }

    private void showMessage(String msg) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show();
    }

    private void openPredictScreen() {
        PredictFragment predictFragment = new PredictFragment();
        FragmentTransaction fragmentTransaction = getSupportFragmentManager().beginTransaction();
        fragmentTransaction.add(R.id.constraint_layout_homepage, predictFragment);
        fragmentTransaction.addToBackStack(null);
        fragmentTransaction.commit();
        predictFragment.setmWatchIP(this.mWatchIP);
    }

    private void openRecordScreen() {
        RecordFragment recordFragment = new RecordFragment();
        FragmentTransaction fragmentTransaction = getSupportFragmentManager().beginTransaction();
        fragmentTransaction.add(R.id.constraint_layout_homepage, recordFragment);
        fragmentTransaction.addToBackStack(null);
        fragmentTransaction.commit();
    }

    private void openRecordScreen2() {
        RecordFragment2 recordFragment2 = new RecordFragment2();
        FragmentTransaction fragmentTransaction2 = getSupportFragmentManager().beginTransaction();
        fragmentTransaction2.add(R.id.constraint_layout_homepage, recordFragment2);
        fragmentTransaction2.addToBackStack(null);
        fragmentTransaction2.commit();
    }

    public ActivityInference getActivityInference() {
        return mActivityInference;
    }

    class ConnectAsyncTask extends AsyncTask<Void, Boolean, Void> {

        private String ipStringStr;
        private int type;

        public ConnectAsyncTask(String ipStringStr, int type) {
            this.ipStringStr = ipStringStr;
            this.type = type;
        }

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            Toast.makeText(MainActivity.this, "Try to connect...", Toast.LENGTH_SHORT).show();
            mProgressBar.setVisibility(View.VISIBLE);
        }

        @Override
        protected Void doInBackground(Void... params) {
            try {
                Socket socket = new Socket(ipStringStr, 5556);
                ObjectOutputStream oos = new ObjectOutputStream(socket.getOutputStream());
                oos.writeObject("Ping!");
                ObjectInputStream ois = new ObjectInputStream(socket.getInputStream());
                String result = (String) ois.readObject();
                if (result.equals("Accepted!")) {
                    publishProgress(true);
                    socket.close();
                } else {
                    publishProgress(false);
                    socket.close();
                }
            } catch (final Exception e) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        showMessage("Error: " + e.getMessage());
                        mProgressBar.setVisibility(View.INVISIBLE);
                    }
                });
            }
            return null;
        }

        @Override
        protected void onProgressUpdate(Boolean... values) {
            super.onProgressUpdate(values);
            mProgressBar.setVisibility(View.INVISIBLE);
            boolean signal = values[0];
            if (signal) {
                if (type == 0) {
                    openPredictScreen();
                } else if(type==1){
                    openRecordScreen();
                }else{
                    openRecordScreen2();
                }
            } else {
                showMessage("Cannot connect!");
            }
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
        }
    }
}
