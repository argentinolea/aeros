#define PF_USE_WRITE 1
#include "MinimumSerial.h"
#include <SensirionI2cScd30.h>
#include <Wire.h>
#include <SPI.h>
//#include <SD.h>
#include "SdFat.h"
SdFat SD;

MinimumSerial MinSerial;
SensirionI2cScd30 sensor;

static char errorMessage[64];
static int16_t error;
const int SD_CHIP_SELECT = 4;
File logFile;
char fileName[30];

void disableI2C() {
  TWCR &= ~(1 << TWEN); // Disable TWI/I2C
  pinMode(SDA, INPUT);  // Release SDA
  pinMode(SCL, INPUT);  // Release SCL
}

void enableI2C() {
  Wire.begin(); // Reinitialize I2C
}

void disableSD() {
  digitalWrite(SD_CHIP_SELECT,HIGH);
}

void enableSD() {
  pinMode(SD_CHIP_SELECT, OUTPUT);
  digitalWrite(SD_CHIP_SELECT,LOW);
}

void writeOnFile(float co2Concentration, float temperature, float humidity) {
  uint32_t fileSize = 0;
  if (!SD.begin(4)) {
      MinSerial.println("SD card initialization failed!");
      //delay(1000); // Allow the user to see the message
      return;      // Exit setup() gracefully
  }
  
  logFile = SD.open(fileName, FILE_WRITE);
  MinSerial.println(logFile);
  if (logFile) {
    logFile.print("MILLIS: ");
    logFile.print(millis());
    logFile.print("\tCO2: ");
    logFile.print(co2Concentration);
    logFile.print("\tTemp: ");
    logFile.print(temperature);
    logFile.print("\tHumidity: ");
    logFile.println(humidity);
    logFile.close();
  } else {
    // if the file didn't open, print an error:
    MinSerial.println("error opening file log");
  }

  fileSize = logFile.fileSize();
  MinSerial.println(fileSize);
  if(fileSize > 10240){
    sprintf(fileName,"co2%lu.log",millis());
  }



  //disableSD();
  //MinSerial.println("Disabling SD");
  //enableI2C();
  //MinSerial.println("Enabled I2C");
}

void setup() {

  MinSerial.begin(115200);
  sprintf(fileName,"co2%lu.log",millis());
  while (!MinSerial) {
    delay(100);
  }
  Wire.begin();
  sensor.begin(Wire, SCD30_I2C_ADDR_61);

  sensor.stopPeriodicMeasurement();
  sensor.softReset();
  delay(2000);
  uint8_t major = 0;
  uint8_t minor = 0;
  error = sensor.readFirmwareVersion(major, minor);
  if (error != NO_ERROR) {
    MinSerial.print(F("Error trying to execute readFirmwareVersion(): "));
    errorToString(error, errorMessage, sizeof errorMessage);
    return;
  }
  /* MinSerial.print("firmware version major: ");
    MinSerial.print(major);
    MinSerial.print("\t");
    MinSerial.print("minor: ");
    MinSerial.print(minor);
    MinSerial.println();*/

  uint16_t isActive = 0;  // Variable to store the status of auto-calibration
  int16_t errorCode;

  MinSerial.println(F("Activating Auto-Calibration..."));
  errorCode = sensor.activateAutoCalibration(1);  // Activate auto-calibration

  if (errorCode == 0) {
    MinSerial.println(F("Auto-Calibration activated successfully."));
  } else {
    MinSerial.print(F("Error activating Auto-Calibration. Error code: "));
    return;  // Exit if activation failed
  }

  // Step 2: Check Auto-Calibration Status
  MinSerial.println(F("Checking Auto-Calibration status..."));
  errorCode = sensor.getAutoCalibrationStatus(isActive);

  if (errorCode == 0) {
    if (isActive) {
      MinSerial.println(F("Auto-Calibration is active."));
    } else {
      MinSerial.println(F("Auto-Calibration is not active."));
    }
  } else {
    MinSerial.print(F("Error retrieving Auto-Calibration status. Error code: "));
  }

  error = sensor.startPeriodicMeasurement(0);
  if (error != NO_ERROR) {
    MinSerial.print(F("Error trying to execute startPeriodicMeasurement(): "));
    return;
  }

}

void loop() {

  float co2Concentration = 0.0;
  float temperature = 0.0;
  float humidity = 0.0;
  delay(5000);
  error = sensor.blockingReadMeasurementData(co2Concentration, temperature,
                                             humidity);
  if (error != NO_ERROR) {
    MinSerial.print("Error trying to execute blockingReadMeasurementData(): ");
    errorToString(error, errorMessage, sizeof errorMessage);
    MinSerial.println(errorMessage);
    return;
  }
  writeOnFile(co2Concentration,temperature,humidity);
  MinSerial.print("CO2: ");
  MinSerial.print(co2Concentration);
  MinSerial.print("\tTemp: ");
  MinSerial.print(temperature);
  MinSerial.print("\tHumidity: ");
  MinSerial.println(humidity);
}
