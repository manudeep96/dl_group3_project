import { Text, Button, Heading, Input, Stack } from '@chakra-ui/react'
import './home.css'
import TextBox from '../../components/textBox'
import { useState } from 'react'

const Home = () => {

    const [text, setText] = useState('')
    const [jobTitle, setJobTitle] = useState('')
    const [speaker, setSpeaker] = useState('')

    const [state, setState] = useState('')

    const [party, setParty] = useState('')

    const [context, setContext] = useState('')

    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const predict = async () => {
        if (!text) {
            return
        }
        setLoading(true)
        const body = { text, jobTitle, speaker, state, party, context }
        const payload = {
            method: "POST", 
            headers: {"Content-Type": "application/json"}, 
            body: JSON.stringify(body)
        }
        try {
            let res = await fetch("http://localhost:5002/api/detect", payload)
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
            <Stack w="80%" margin="40px" spacing={8} padding={'8px'}>
                <TextBox text={text} setText={setText} />
                <Input value={jobTitle} onChange={(e) => setJobTitle(e.target.value)} background={'blue.50'} placeholder='Job Title' />
                <Input value={speaker} onChange={(e) => setSpeaker(e.target.value)} background={'blue.50'} placeholder='Speaker' />
                <Input value={state} onChange={(e) => setState(e.target.value)} background={'blue.50'} placeholder='State' />
                <Input value={party} onChange={(e) => setParty(e.target.value)} background={'blue.50'} placeholder='Party affiliation' />
                <Input value={context} onChange={(e) => setContext(e.target.value)} background={'blue.50'} placeholder='Context' />
            </Stack>
            <Button isLoading={loading} onClick={() => predict()} colorScheme='blue'> Predict correctness </ Button>
            {result ? (<Text as='b' fontSize={'xl'} marginTop={4} color={'blue.50'}>{`This text is likely to be ${result}`}</Text>) : <></>}
        </div>)
}

export default Home