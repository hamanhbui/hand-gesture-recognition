package com.example.cgr;


import android.content.Context;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Environment;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;

import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.components.Legend;
import com.github.mikephil.charting.components.XAxis;
import com.github.mikephil.charting.components.YAxis;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet;

import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/**
 * A simple {@link Fragment} subclass.
 */
public class RecordFragment2 extends Fragment {

    private float x = 0, y = 0, z = 0, x1 = 0, y1 = 0, z1 = 0,x2=0,y2=0,z2=0;
    private Thread visualizeAcceThread;
    private BufferedWriter outputWriter;
    private String folderName;
    private LineChart mChart;
    private Spinner mSpinner;
    private MainActivity mActivity;
    private Button mFinish;
    private Button unknownBtn;
    private Button zeroBtn;
    private Button oneBtn;
    private Button twoBtn;
    private Button threeBtn;
    private Button fourBtn;
    private Button fiveBtn;
    private Button sixBtn;
    private Button sevenBtn;
    private Button eightBtn;
    private Button startGestureBtn;
    private Button nineBtn;
    private Button cWCircleBtn;
    private Button cCWCircleBtn;
    public RecordFragment2() {
        // Required empty public constructor
    }


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_record2, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        initViews(view);
        createFile();
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                startStreaming();
            }
        });
        thread.start();
    }

    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        mActivity = (MainActivity) context;
    }

    private void createFile() {
        try {
            folderName = "Recorder_" + new SimpleDateFormat("yyyy_MM_dd_HH_mm").format(new Date());
            String filepath = Environment.getExternalStorageDirectory().getPath();
            File file = new File(filepath, folderName);
            if (!file.exists()) {
                file.mkdirs();
            }
            String fileName = file.getAbsolutePath() + "/" + "data.txt";
            outputWriter = new BufferedWriter(new FileWriter(fileName));
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    private void initViews(View view) {
        mChart = view.findViewById(R.id.chart1);
        mFinish=view.findViewById(R.id.button4);
        mSpinner = view.findViewById(R.id.spinner);

        unknownBtn =view.findViewById(R.id.button6);
        zeroBtn=view.findViewById(R.id.button14);
        oneBtn=view.findViewById(R.id.button10);
        twoBtn=view.findViewById(R.id.button9);
        threeBtn=view.findViewById(R.id.button2);
        fourBtn=view.findViewById(R.id.button3);
        fiveBtn=view.findViewById(R.id.button12);
        sixBtn=view.findViewById(R.id.button8);
        sevenBtn=view.findViewById(R.id.button11);
        eightBtn=view.findViewById(R.id.button13);
        startGestureBtn=view.findViewById(R.id.button7);
        nineBtn=view.findViewById(R.id.button15);
        cWCircleBtn=view.findViewById(R.id.button);
        cCWCircleBtn=view.findViewById(R.id.button5);

        mFinish.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    outputWriter.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });
        unknownBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    mSpinner.setSelection(0);
                } catch (Exception e) {

                }
            }
        });
        cWCircleBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    mSpinner.setSelection(2);
                } catch (Exception e) {

                }
            }
        });
        cCWCircleBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    mSpinner.setSelection(3);
                } catch (Exception e) {

                }
            }
        });
        zeroBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    mSpinner.setSelection(4);
                } catch (Exception e) {

                }
            }
        });
        oneBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    mSpinner.setSelection(5);
                } catch (Exception e) {

                }
            }
        });
        twoBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    mSpinner.setSelection(6);
                } catch (Exception e) {

                }
            }
        });
        threeBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    mSpinner.setSelection(7);
                } catch (Exception e) {

                }
            }
        });
        eightBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    mSpinner.setSelection(12);
                } catch (Exception e) {

                }
            }
        });
        fourBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    mSpinner.setSelection(8);
                } catch (Exception e) {

                }
            }
        });
        fiveBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    mSpinner.setSelection(9);
                } catch (Exception e) {

                }
            }
        });
        sixBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    mSpinner.setSelection(10);
                } catch (Exception e) {

                }
            }
        });
        sevenBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    mSpinner.setSelection(11);
                } catch (Exception e) {

                }
            }
        });
        startGestureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    mSpinner.setSelection(1);
                } catch (Exception e) {

                }
            }
        });
        nineBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    mSpinner.setSelection(13);
                } catch (Exception e) {

                }
            }
        });
        addItemsOnSpinner();
        acceGraphInit();
    }
    private void startStreaming() {
        try {
            DatagramSocket socket = new DatagramSocket(5556);
            feedMultiple();
            byte[] buffer = new byte[2048];
            while (true) {
                DatagramPacket receivePacket = new DatagramPacket(buffer, buffer.length);
                socket.receive(receivePacket);
                ByteArrayInputStream bais = new ByteArrayInputStream(buffer);
                ObjectInputStream ois = new ObjectInputStream(bais);
                ArrayList<Object> listData = (ArrayList<Object>) ois.readObject();
                for (int i = 0; i < listData.size(); i += 2) {
                    String currentDateandTime = (String) listData.get(i);
                    float[] a = (float[]) listData.get(i + 1);
                    x = a[0];
                    y = a[1];
                    z = a[2];
                    x1 = a[3];
                    y1 = a[4];
                    z1 = a[5];
                    x2 = a[6];
                    y2 = a[7];
                    z2 = a[8];
                    outputWriter.write(currentDateandTime + "," + x + "," + y + "," + z + "," +
                            x1 + "," + y1 + "," + z1 + "," +x2 + "," + y2 + "," + z2 + "," +
                            mSpinner.getSelectedItem().toString() + ";" + "\r\n");
                    long startTime2 = System.currentTimeMillis();
                    long currentTime2 = startTime2;
                    while (currentTime2 < startTime2 + 35) {
                        currentTime2 = System.currentTimeMillis();
                    }
                }
            }
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    private void addItemsOnSpinner() {
        List<String> list = new ArrayList<String>();
        list.add("Unknown");
        list.add("Start_gesture");
        list.add("CWCircle");
        list.add("CCWCircle");
        list.add("0");
        list.add("1");
        list.add("2");
        list.add("3");
        list.add("4");
        list.add("5");
        list.add("6");
        list.add("7");
        list.add("8");
        list.add("9");
        ArrayAdapter<String> dataAdapter = new ArrayAdapter<>(getActivity(),
                android.R.layout.simple_spinner_item, list);
        dataAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        mSpinner.setAdapter(dataAdapter);
    }

    private void acceGraphInit() {
        // enable description text
        mChart.getDescription().setEnabled(true);
        mChart.setContentDescription("Accelerometer Legenda");

        // set an alternative background color
        mChart.setBackgroundColor(Color.LTGRAY);

        LineData data = new LineData();
        data.setValueTextColor(Color.WHITE);

        // add empty data
        mChart.setData(data);
        mChart.getDescription().setEnabled(false);

        // get the legend (only possible after setting data)
        Legend l = mChart.getLegend();

        // modify the legend ...
        l.setForm(Legend.LegendForm.LINE);
        l.setTextColor(Color.WHITE);

        XAxis xl = mChart.getXAxis();
        xl.setTextColor(Color.WHITE);
        xl.setDrawGridLines(false);
        xl.setAvoidFirstLastClipping(true);
        xl.setEnabled(false);

        YAxis leftAxis = mChart.getAxisLeft();
        leftAxis.setTextColor(Color.WHITE);
        leftAxis.setAxisMaximum(20f);
        leftAxis.setAxisMinimum(-20f);
        leftAxis.setDrawGridLines(true);

        YAxis rightAxis = mChart.getAxisRight();
        rightAxis.setEnabled(false);
    }

    private void feedMultiple() {
        final Runnable runnable = new Runnable() {
            @Override
            public void run() {
                addEntry();
            }
        };

        visualizeAcceThread = new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    getActivity().runOnUiThread(runnable);
                    try {
                        Thread.sleep(25);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        });

        visualizeAcceThread.start();
    }

    private void addEntry() {
        LineData data = mChart.getData();

        if (data != null) {
            ILineDataSet setX = data.getDataSetByIndex(0);
            ILineDataSet setY = data.getDataSetByIndex(1);
            ILineDataSet setZ = data.getDataSetByIndex(2);
            ILineDataSet setX1 = data.getDataSetByIndex(3);
            ILineDataSet setY1 = data.getDataSetByIndex(4);
            ILineDataSet setZ1 = data.getDataSetByIndex(5);

            if (setX == null || setY == null || setZ == null) {
                setX = createSet("Acce X", Color.BLUE);
                setY = createSet("Acce Y", Color.RED);

                setZ = createSet("Acce Z", Color.GREEN);
                setX1 = createSet("Gyro X", Color.GRAY);
                setY1 = createSet("Gyro Y", Color.BLACK);

                setZ1 = createSet("Gyro Z", Color.YELLOW);
                data.addDataSet(setX);
                data.addDataSet(setY);
                data.addDataSet(setZ);
                data.addDataSet(setX1);
                data.addDataSet(setY1);
                data.addDataSet(setZ1);
            }

            data.addEntry(new Entry(setX.getEntryCount(), x), 0);
            data.addEntry(new Entry(setY.getEntryCount(), y), 1);
            data.addEntry(new Entry(setZ.getEntryCount(), z), 2);
            data.addEntry(new Entry(setX1.getEntryCount(), x1), 3);
            data.addEntry(new Entry(setY1.getEntryCount(), y1), 4);
            data.addEntry(new Entry(setZ1.getEntryCount(), z1), 5);

            data.notifyDataChanged();

            mChart.notifyDataSetChanged();
            mChart.setVisibleXRangeMaximum(120);
            mChart.moveViewToX(data.getEntryCount());
        }
    }

    private LineDataSet createSet(String labelName, int colorValue) {
        LineDataSet set = new LineDataSet(null, labelName);
        set.setColor(colorValue);
        set.setDrawCircles(false);
        set.setDrawValues(false);
        return set;
    }
}
