const express = require('express');
const detectRoutes = require('./routes/detectRoutes');
const cors=require("cors");
const errorHandler = require('./middleware/errorMiddleware')

const app = express();
const PORT = process.env.PORT || 5002;

const corsOptions ={
    origin:'*', 
    credentials:true,            //access-control-allow-credentials:true
    optionSuccessStatus:200,
 }

app.use(cors(corsOptions))
app.use(express.urlencoded({extended: true}));
app.use(express.json());

app.use("/api/detect", detectRoutes);

app.use(errorHandler);

app.listen(PORT, (error) => {
    if (!error) {
        console.log(`Listening on port ${PORT}`)
    } else {
        console.log(error)
    }
})