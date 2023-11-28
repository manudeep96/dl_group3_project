import { Box, Button, Heading , Text} from '@chakra-ui/react'
import './home.css'
import TextBox from '../../components/textBox'
import { useState } from 'react'

const Home = () => {

    const [text, setText] = useState('')
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const predict = async () => {
        if (!text) {
            return
        }
        setLoading(true)
        const body = { text: text }
        const payload = { method: "POST", body: JSON.stringify(body) }
        try {
            let res = await fetch("http://localhost:5000/api/detect", payload)
            res = await res.json();
            setResult(res?.data)
        } catch (e) {
            console.log(e);
        }
        setLoading(false)
    }

    return (
        <div className='container'>
            <Heading size='xl' marginTop={4} color={'blue.50'}>
                Fake news detector
            </Heading>
            <TextBox text={text} setText={setText} />
            <Button isLoading={loading} onClick={() => predict()} colorScheme='blue'> Predict correctness </ Button>
            {result? (<Text as='b' fontSize={'xl'} marginTop={4} color={'blue.50'}>{`This text is likely to be ${result}`}</Text>): <></>}
        </div>)
}

export default Home