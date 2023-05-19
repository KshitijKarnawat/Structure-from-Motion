// int motor1pin1 = 2;
// int motor1pin2 = 4;
// int motor1pwm = 3;

// int motor2pin1 = 6;
// int motor2pin2 = 7;
// int motor2pwm = 5;
const int lf = 11;
const int lb = 10;
const int rb = 9;
const int rf = 6;
void setup() {
  Serial.begin(115200);
//  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(lf, OUTPUT);
  pinMode(lb, OUTPUT);
  pinMode(rb, OUTPUT);
  pinMode(rf, OUTPUT);
}

String rx_data;
char cmd;
int left_wh_speed = 0;
int right_wh_speed = 0;
char data_read[3];

void loop() {
  if(Serial.available())
  {
    Serial.readBytes(data_read,3);
//    cmd = Serial.read();
    cmd = data_read[0];
//    left_wh_speed = Serial.read();
    left_wh_speed = data_read[1];
//    right_wh_speed = Serial.read();
    right_wh_speed = data_read[2];
//    Serial.println(cmd);
//    Serial.println(left_wh_speed);
//    Serial.println(right_wh_speed);

    
    left_wh_speed = left_wh_speed -63;
    right_wh_speed = right_wh_speed -63;

    if(left_wh_speed <= 0)
    {
      analogWrite(lb,(left_wh_speed+63)*4);
      analogWrite(lf, 0);
    }
    else
    {
      analogWrite(lf,(left_wh_speed+63)*4);
      analogWrite(lb, 0);
    }
  
  
    if(right_wh_speed <= 0)
    {
      analogWrite(rb,(right_wh_speed+63)*4);
      analogWrite(rf, 0);
    }
    else
    {
      analogWrite(rf,(right_wh_speed+63)*4);
      analogWrite(rb, 0);
    }
    
//    if(left_wh_speed == 121)
//    {
//      digitalWrite(LED_BUILTIN, HIGH);
//      delay(500);  
//    }
//    digitalWrite(LED_BUILTIN, LOW);
//    delay(500); 
//    
  }
}
  