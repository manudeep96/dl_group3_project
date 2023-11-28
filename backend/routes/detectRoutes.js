const express = require('express');
const router = express.Router();

const {detectClass} = require('../controllers/detectController');

router.route('/').post(detectClass);

module.exports = router