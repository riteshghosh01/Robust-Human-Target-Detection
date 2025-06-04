const mongoose = require('mongoose');

const mongoURL = 'mongodb://localhost:27017/food_ordering_app'; 
// Replace with your MongoDB connection string
mongoose.connect(mongoURL, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

const db = mongoose.connection;
db.on('connected', () => {
  console.log('Connected to MongoDB database');
});

db.on('error', (error) => {
  console.error('Error connecting to MongoDB:', error);
});
db.on('disconnected', () => {
  console.log('Disconnected from MongoDB database');
});    
module.exports = db;