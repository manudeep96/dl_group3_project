expressAsyncHandler = require('express-async-handler')
const {spawn} = require('child_process');

const detectClass = expressAsyncHandler( async(req, res) => {
    console.log(req.body);
    // Call python script
    const pyScript = spawn('python', ['./scripts/predictor.py', req.body.text,req.body.speaker,req.body.jobTitle,req.body.state,req.body.party])
    pyScript.stdout.on('data', (data) => {
        console.log(`Output ${data}`)
        data = data.toString().replace("\r\n", "");
        res.setHeader('Access-Control-Allow-Origin', '*');
        res.status(200).json({message: `Successfully detected class of the text`, data})
    })
})

module.exports = {
    detectClass
}