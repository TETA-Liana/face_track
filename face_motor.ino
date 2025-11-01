#include <Stepper.h>

const int stepsPerRevolution = 2048;  // 28BYJ-48 motor
Stepper myStepper(stepsPerRevolution, 8, 10, 9, 11);

void setup() {
  Serial.begin(9600);
  myStepper.setSpeed(10);  // 10 RPM
  Serial.println("Ready");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();

    if (cmd == 'L') {
      Serial.println("Turning Left");
      myStepper.step(-100);  // rotate counterclockwise
    } 
    else if (cmd == 'R') {
      Serial.println("Turning Right");
      myStepper.step(100);   // rotate clockwise
    }
  }
}
