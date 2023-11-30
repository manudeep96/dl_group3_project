import './App.css';
import Home from './views/home/home';
import { Box, ChakraProvider } from '@chakra-ui/react';

function App() {
  return (
    <ChakraProvider>
      <Box minH={'100vh'} h={'100%'} padding={'5%'} background={'gray.900'}>
        <Home />
      </Box>
    </ChakraProvider>
  );
}

export default App;
