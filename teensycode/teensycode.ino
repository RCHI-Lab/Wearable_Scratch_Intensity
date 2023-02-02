
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <SPI.h>
#include <ADXL362.h>
#define SCB_AIRCR (*(volatile uint32_t *)0xE000ED0C)

IntervalTimer myTimer1;
IntervalTimer myTimer2; 

int PIEZO_PIN = 16;
int piezoADC; 
int buttonpin = 0; 
int samp_freq1 = 8000; 
int samp_freq2 = 400; 

byte iden1 = B00000001;
byte iden2 = B00000011;
char newline = B11111111;

int16_t XValue, YValue, ZValue, Temperature;
unsigned long curr_time; 
unsigned long start_time; 
int signx = 0; 
int signy = 0; 
int signz = 0; 
int buttonpress = 0; 



ADXL362 xl;

void setup(void) {
  Serial.begin(115200);
  xl.begin(10);                   // Setup SPI protocol, issue device soft reset
  xl.beginMeasure();              // Switch ADXL362 to measure mode  

  pinMode(buttonpin, INPUT); 
  
  myTimer1.priority(255); 
  myTimer2.priority(255); 
  
  start_time = micros();
  
  //delete when button is added  
  myTimer1.begin(con, 1000000/samp_freq1);
  myTimer2.begin(acc, 1000000/samp_freq2); 
}

void con() {
  curr_time = micros() - start_time;
  piezoADC = analogRead(PIEZO_PIN);
  Serial.write((byte *)&newline, 1); 
  Serial.write((byte *)&newline, 1); 
  Serial.write(iden1);
  Serial.write((byte *)&piezoADC, 2);
  Serial.write((byte *)&curr_time, 4);
  Serial.flush();
}

void acc() {
  curr_time = micros() - start_time;
  xl.readXYZTData(XValue, YValue, ZValue, Temperature);
  signx = 1;
  signy = 1; 
  signz = 1; 
  if (XValue < 0 ) 
  {
    signx = 0;
  }
  if (YValue < 0 ) 
  {
    signy = 0; 
  }
  if (ZValue < 0 ) 
  {
    signz = 0; 
  }
  XValue = abs(XValue); 
  YValue = abs(YValue); 
  ZValue = abs(ZValue);
  Serial.write((byte *)&newline, 1); 
  Serial.write((byte *)&newline, 1); 
  Serial.write(iden2);
  Serial.write((byte *)&XValue, 2); 
  Serial.write((byte *)&YValue, 2); 
  Serial.write((byte *)&ZValue, 2); 
  Serial.write((byte *)&curr_time, 4);
  Serial.write((byte *)&signx, 1); 
  Serial.write((byte *)&signy, 1); 
  Serial.write((byte *)&signz, 1); 
  Serial.flush();
}

void loop() {
  /*
  buttonpress = digitalRead(buttonpin); 
  
  if (buttonpress == 1)  {
      myTimer1.end();
      myTimer3.end();
      Serial.write(newline); 
      Serial.write(newline); 
      Serial.write(newline); 
      Serial.write(newline); 
      Serial.flush();
      delay(2000);
      start_time = micros(); 
      myTimer1.begin(con, 1000000/samp_freq1);
      myTimer2.begin(acc, 1000000/samp_freq3); 

  }
  */

}
