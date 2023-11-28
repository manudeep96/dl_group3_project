const express = require('express');
const detectRoutes = require('./routes/detectRoutes');
const app = express();
const PORT = process.env.PORT || 5000;
const errorHandler = require('./middleware/errorMiddleware')
app.use(express.json());
app.use(express.urlencoded({extended: false}));

app.use("/api/detect", detectRoutes);
app.use(errorHandler);

app.listen(PORT, (error) => {
    if (!error) {
        console.log(`Listening on port ${PORT}`)
    } else {
        console.log(error)
    }
})