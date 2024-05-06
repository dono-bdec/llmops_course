class Vehicle:
    def __init__(self, name, brand, seats, gear_type, cost):
        self.name = name
        self.brand = brand
        self.seats = seats
        self.gear_type = gear_type
        self.cost = cost
    
    def getVehicleDetails(self):
        return f"The car {self.name} from {self.brand} seats {self.seats} people and comes with {self.gear_type} transmission at a cost of {self.cost}"

myCars =[["Swift", "Maruti", 5, "Automatic", "10 lakhs"],
       ["Baleno", "Maruti", 3, "Manual", "15 lakhs"],
       ["Creta", "Hyundai", 7, "Manual", "12 lakhs"],
       ["Verna", "Hyundai", 4, "Automatic", "18 lakhs"],
       ["XUV 300", "Tata", 8, "Automatic", "24 lakhs"],
       ["Altroz", "Tata", 4, "Automatic", "14 lakhs"],
       ["Sunny", "Nissan", 5, "Manual", "30 lakhs"]]


for i in range(0,len(myCars)):
    myVehicle = Vehicle(myCars[i][0],myCars[i][1],myCars[i][2],myCars[i][3],myCars[i][4])
    print(myVehicle.getVehicleDetails())